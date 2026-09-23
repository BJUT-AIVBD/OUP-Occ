#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mmcv
import mmdet
import numpy as np
import torch
from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.datasets.occ_metrics import Metric_mIoU
from mmdet3d.models.occ_uncertainty import align_logits_to_bxyzc, align_tensor_to_bxyz
from tools.analyze_occ_uncertainty import (
    build_valid_mask,
    distance_masks,
    get_img_metas,
    get_ignore_index,
    get_num_classes,
    patch_analysis_pipeline,
    require_batch_tensor,
    sample_name_from_meta,
    tensor_to_numpy,
)
from tools.analyze_refinement_effect import (
    add_official_metric,
    build_eval_model,
    per_class_iou,
    safe_name,
    safe_rate,
    setup_logger,
    top_uncertainty_mask,
)

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ATUM effect')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_atum_fulltrain_analysis_full_val')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--full-val', action='store_true')
    parser.add_argument('--vis-topk', type=int, default=30)
    parser.add_argument('--eval-mask', default='camera', choices=['camera', 'lidar', 'both', 'union', 'all'])
    parser.add_argument('--official-style-metric', action='store_true', default=True)
    parser.add_argument('--custom-metric', dest='official_style_metric', action='store_false')
    parser.add_argument('--ignore-index', type=int, default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=True)


def bev_error(mask):
    return mask.max(axis=2).astype(float)


def bev_semantic(label, free_class=17):
    occupied = label != free_class
    if not occupied.any():
        return np.full(label.shape[:2], free_class, dtype=np.int64)
    z_ids = np.arange(label.shape[2]).reshape(1, 1, -1)
    score = np.where(occupied, z_ids, -1)
    top_z = score.argmax(axis=2)
    bev = np.take_along_axis(label, top_z[..., None], axis=2).squeeze(2)
    bev[score.max(axis=2) < 0] = free_class
    return bev


def miou_from_metric(metric):
    iou = per_class_iou(metric.hist)
    return float(np.nanmean(iou[:-1]) * 100.0), iou * 100.0


def update_region_stats(store, name, region, tuned_err, atum_err, u_enhanced=None):
    count = int(region.sum())
    store[name]['voxel_count'] += count
    store[name]['tuned_errors'] += int(tuned_err[region].sum())
    store[name]['atum_errors'] += int(atum_err[region].sum())
    if u_enhanced is not None:
        store[name]['uncertainty_sum'] += float(u_enhanced[region].sum())


def finalize_region_stats(store):
    out = {}
    rows = []
    for name, stats in store.items():
        count = stats['voxel_count']
        tuned = stats['tuned_errors'] / count if count else float('nan')
        atum = stats['atum_errors'] / count if count else float('nan')
        item = dict(
            voxel_count=count,
            tuned_error_rate=tuned,
            atum_error_rate=atum,
            error_reduction=tuned - atum if count else float('nan'),
            mean_enhanced_uncertainty=stats['uncertainty_sum'] / count if count else float('nan'))
        out[name] = item
        rows.append(dict(region=name, **item))
    return out, rows


def save_panel(path, title_items, cmaps=None, vmins=None, vmaxs=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cmaps = cmaps or ['viridis'] * 4
    vmins = vmins or [None] * 4
    vmaxs = vmaxs or [None] * 4
    for ax, (title, arr), cmap, vmin, vmax in zip(axes.flat, title_items, cmaps, vmins, vmaxs):
        im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def update_memory_acc(acc, current, memory, enhanced, gate, valid_mask):
    acc['count'] += 1
    acc['current_sum'] += float(current.mean())
    acc['memory_sum'] += float(memory.mean())
    acc['enhanced_sum'] += float(enhanced.mean())
    acc['gate_sum'] += float(gate.mean())
    acc['valid_ratio_sum'] += float(valid_mask.mean())
    acc['current_var_sum'] += float(np.var(current))
    acc['enhanced_var_sum'] += float(np.var(enhanced))


def finalize_memory_acc(acc, cfg):
    count = max(1, acc['count'])
    current_var = acc['current_var_sum'] / count
    enhanced_var = acc['enhanced_var_sum'] / count
    return dict(
        num_memory_scenes=int(acc.get('num_memory_scenes', 0)),
        mean_memory_value=acc['memory_sum'] / count,
        mean_current_uncertainty=acc['current_sum'] / count,
        mean_enhanced_uncertainty=acc['enhanced_sum'] / count,
        mean_memory_gate=acc['gate_sum'] / count,
        memory_valid_ratio=acc['valid_ratio_sum'] / count,
        train_memory_mode=cfg.model.temporal_memory.train_memory_mode,
        test_memory_mode=cfg.model.temporal_memory.test_memory_mode,
        current_uncertainty_variance=current_var,
        enhanced_uncertainty_variance=enhanced_var,
        variance_reduction=current_var - enhanced_var)


def write_readme(out_dir):
    with open(os.path.join(out_dir, 'README_ANALYSIS.md'), 'w') as f:
        f.write(
            '# ATUM Analysis\n\n'
            'This directory compares OUP-Occ tuned 52.13 with OUP-Occ + ATUM. '
            'ATUM stores BEV uncertainty memory only and uses enhanced uncertainty '
            'to guide progressive refinement.\n')


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    vis_dir = os.path.join(args.out_dir, 'vis')
    mmcv.mkdir_or_exist(vis_dir)
    logger = setup_logger(args.out_dir)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    baseline_cfg = compat_cfg(Config.fromfile(args.baseline_config))
    setup_multi_processes(cfg)
    set_random_seed(args.seed, deterministic=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for ATUM analysis.')

    dataset_cfg = patch_analysis_pipeline(cfg.data[args.split])
    dataset = build_dataset(dataset_cfg)
    full_val_size = len(dataset)
    sample_limit = None if args.full_val or args.max_samples is None or args.max_samples < 0 else args.max_samples
    if sample_limit is not None and sample_limit >= full_val_size:
        sample_limit = None
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)

    num_classes = get_num_classes(cfg)
    ignore_index = get_ignore_index(cfg, args)
    class_names = Metric_mIoU(num_classes=num_classes).class_names[:num_classes]
    tuned_model = build_eval_model(baseline_cfg, args.baseline_ckpt, args.gpu_id, dataset, logger, 'tuned')
    atum_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'atum')
    if hasattr(atum_model.module, 'reset_temporal_memory'):
        atum_model.module.reset_temporal_memory()

    metric_tuned = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    metric_atum = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    class_gt_counts = np.zeros(num_classes, dtype=np.float64)
    stat_factory = lambda: {
        'voxel_count': 0,
        'tuned_errors': 0,
        'atum_errors': 0,
        'uncertainty_sum': 0.0
    }
    high_stats = defaultdict(stat_factory)
    visibility_stats = defaultdict(stat_factory)
    distance_stats_acc = defaultdict(stat_factory)
    memory_acc = defaultdict(float)
    prev_scene_maps = {}
    processed = 0

    for _, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            tuned_out = tuned_model(return_loss=True, return_result=True, **data_batch)
            atum_out = atum_model(return_loss=True, return_result=True, **data_batch)

        tuned_logits = align_logits_to_bxyzc(
            tuned_out.get('output_occ_logits', tuned_out['final_occ_logits']), num_classes=num_classes)
        atum_logits = align_logits_to_bxyzc(
            atum_out.get('output_occ_logits', atum_out['final_occ_logits']), num_classes=num_classes)
        spatial_shape = tuple(atum_logits.shape[1:4])
        gt = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'voxel_semantics'), spatial_shape, 'voxel_semantics')
        mask_camera = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_camera'), spatial_shape, 'mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_lidar'), spatial_shape, 'mask_lidar').bool()
        pred_tuned = tuned_logits.softmax(-1).argmax(-1)
        pred_atum = atum_logits.softmax(-1).argmax(-1)
        unc = atum_out['uncertainty_dict']
        u_current = unc.get('U_current_bev', unc['U_total_bev_feat'])
        u_memory = unc.get('U_memory_bev', u_current)
        u_enhanced = unc.get('U_enhanced_bev', unc.get('U_refinement_bev_feat', u_current))
        memory_gate = unc.get('memory_gate', torch.zeros_like(u_current))
        memory_valid = unc.get('memory_valid_mask', torch.zeros_like(u_current))
        gate_map = unc.get('gate_map', u_enhanced)

        metas = get_img_metas(data_batch)
        batch_size = int(atum_logits.shape[0])
        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            meta = metas[b] if b < len(metas) else {}
            sample_name = sample_name_from_meta(meta, processed)
            file_name = safe_name(sample_name)
            scene_token = str(meta.get('scene_token', 'unknown')) if isinstance(meta, dict) else 'unknown'

            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            tuned_np = tensor_to_numpy(pred_tuned[b]).astype(np.int16)
            atum_np = tensor_to_numpy(pred_atum[b]).astype(np.int16)
            current_np = tensor_to_numpy(u_current[b, 0]).astype(np.float32)
            memory_np = tensor_to_numpy(u_memory[b, 0]).astype(np.float32)
            enhanced_np = tensor_to_numpy(u_enhanced[b, 0]).astype(np.float32)
            gate_np = tensor_to_numpy(memory_gate[b, 0]).astype(np.float32)
            valid_np = tensor_to_numpy(memory_valid[b, 0]).astype(np.float32)
            gate_map_np = tensor_to_numpy(gate_map[b]).astype(np.float32)
            if gate_map_np.ndim == 3:
                gate_map_np = gate_map_np.mean(axis=0)

            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            add_official_metric(metric_tuned, tuned_np, gt_np, mask_lidar_np, mask_camera_np)
            add_official_metric(metric_atum, atum_np, gt_np, mask_lidar_np, mask_camera_np)
            metric_valid = mask_camera_np & valid_label
            for cls_idx in range(num_classes):
                class_gt_counts[cls_idx] += int(((gt_np == cls_idx) & metric_valid).sum())

            tuned_err = (tuned_np != gt_np) & valid_mask
            atum_err = (atum_np != gt_np) & valid_mask
            tuned_label_err = (tuned_np != gt_np) & valid_label
            atum_label_err = (atum_np != gt_np) & valid_label
            corrected = tuned_err & ~atum_err
            degraded = ~tuned_err & atum_err & valid_mask
            enhanced_voxel = np.repeat(enhanced_np.T[:, :, None], gt_np.shape[2], axis=2)

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(enhanced_voxel, valid_mask, ratio)
                update_region_stats(high_stats, key, region, tuned_err, atum_err, enhanced_voxel)

            regions = {
                'visible_by_both': mask_camera_np & mask_lidar_np,
                'camera_only': mask_camera_np & ~mask_lidar_np,
                'lidar_only': ~mask_camera_np & mask_lidar_np,
                'invisible_by_both': ~mask_camera_np & ~mask_lidar_np,
            }
            for name, region in regions.items():
                update_region_stats(
                    visibility_stats, name, region & valid_label,
                    tuned_label_err, atum_label_err, enhanced_voxel)

            dist_regions = distance_masks(gt_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    update_region_stats(
                        distance_stats_acc, name, region & valid_mask,
                        tuned_err, atum_err, enhanced_voxel)

            update_memory_acc(memory_acc, current_np, memory_np, enhanced_np, gate_np, valid_np)

            if processed < args.vis_topk:
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_uncertainty_memory_comparison.png'),
                    [
                        ('U_current_bev', current_np),
                        ('U_memory_bev', memory_np),
                        ('U_enhanced_bev', enhanced_np),
                        ('memory_gate', gate_np),
                    ],
                    ['viridis', 'viridis', 'viridis', 'viridis'],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_tuned_vs_atum_error_overlay.png'),
                    [
                        ('tuned error', bev_error(tuned_err)),
                        ('ATUM error', bev_error(atum_err)),
                        ('improved voxels', bev_error(corrected)),
                        ('degraded voxels', bev_error(degraded)),
                    ],
                    ['Reds', 'Reds', 'Greens', 'Reds'],
                    [0.0] * 4,
                    [1.0] * 4)
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_refinement_gate_before_after_memory.png'),
                    [
                        ('current uncertainty guidance', current_np),
                        ('enhanced uncertainty guidance', enhanced_np),
                        ('enhanced-current difference', enhanced_np - current_np),
                        ('corrected voxels', bev_error(corrected)),
                    ],
                    ['viridis', 'viridis', 'coolwarm', 'Greens'])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_semantic_comparison.png'),
                    [
                        ('GT BEV semantic', bev_semantic(gt_np, num_classes - 1)),
                        ('OUP tuned prediction', bev_semantic(tuned_np, num_classes - 1)),
                        ('ATUM prediction', bev_semantic(atum_np, num_classes - 1)),
                        ('improved regions', bev_error(corrected)),
                    ],
                    ['tab20', 'tab20', 'tab20', 'Greens'])
                if scene_token in prev_scene_maps:
                    prev_current = prev_scene_maps[scene_token]
                    save_panel(
                        os.path.join(vis_dir, f'{file_name}_temporal_sequence_uncertainty.png'),
                        [
                            ('U_current previous frame', prev_current),
                            ('U_current current frame', current_np),
                            ('U_memory current frame', memory_np),
                            ('U_enhanced current frame', enhanced_np),
                        ],
                        ['viridis'] * 4,
                        [0.0] * 4,
                        [1.0] * 4)
            prev_scene_maps[scene_token] = current_np
            processed += 1
            logger.info('Processed %s', sample_name)

    if hasattr(atum_model.module, 'temporal_memory') and atum_model.module.temporal_memory is not None:
        memory_acc['num_memory_scenes'] = len(atum_model.module.temporal_memory.scene_memory)
    tuned_miou, tuned_iou = miou_from_metric(metric_tuned)
    atum_miou, atum_iou = miou_from_metric(metric_atum)
    high_summary, high_rows = finalize_region_stats(high_stats)
    visibility_summary, visibility_rows = finalize_region_stats(visibility_stats)
    distance_summary, distance_rows = finalize_region_stats(distance_stats_acc)
    memory_stats = finalize_memory_acc(memory_acc, cfg)

    per_class_rows = []
    for cls_idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=cls_idx,
            class_name=class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            num_gt_voxels=int(class_gt_counts[cls_idx]),
            tuned_iou=float(tuned_iou[cls_idx]),
            atum_iou=float(atum_iou[cls_idx]),
            atum_minus_tuned=float(atum_iou[cls_idx] - tuned_iou[cls_idx])))

    temporal_stability = dict(
        current_uncertainty_variance=memory_stats['current_uncertainty_variance'],
        enhanced_uncertainty_variance=memory_stats['enhanced_uncertainty_variance'],
        variance_reduction=memory_stats['variance_reduction'],
        mean_memory_gate=memory_stats['mean_memory_gate'])
    summary = dict(
        overall=dict(
            tuned_mIoU=tuned_miou,
            atum_mIoU=atum_miou,
            atum_minus_tuned=atum_miou - tuned_miou),
        evaluation_protocol=dict(
            num_samples=processed,
            full_val_size=full_val_size,
            is_full_val=(processed == full_val_size),
            metric_style='official' if args.official_style_metric else 'custom',
            eval_mask=args.eval_mask,
            include_free_in_mIoU=False,
            class_names=class_names),
        high_uncertainty_regions=high_summary,
        temporal_stability=temporal_stability,
        visibility_regions=visibility_summary,
        distance_regions=distance_summary,
        per_class_top_improvements=sorted(
            [
                row for row in per_class_rows
                if row['class_index'] != num_classes - 1
                and not math.isnan(row['atum_minus_tuned'])
            ],
            key=lambda x: x['atum_minus_tuned'],
            reverse=True)[:10],
        memory_statistics=memory_stats)

    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(os.path.join(args.out_dir, 'high_uncertainty_improvement.csv'), high_rows)
    write_csv(os.path.join(args.out_dir, 'visibility_region_improvement.csv'), visibility_rows)
    write_csv(os.path.join(args.out_dir, 'distance_region_improvement.csv'), distance_rows)
    write_csv(os.path.join(args.out_dir, 'temporal_stability.csv'), [temporal_stability])
    write_csv(os.path.join(args.out_dir, 'memory_statistics.csv'), [memory_stats])
    write_readme(args.out_dir)
    logger.info('Saved ATUM analysis to %s', args.out_dir)
    print('\n==== ATUM Analysis Summary ====')
    print(f'   - tuned mIoU: {tuned_miou:.4f}')
    print(f'   - ATUM mIoU: {atum_miou:.4f}')
    print(f'   - ATUM - tuned: {atum_miou - tuned_miou:.4f}')


if __name__ == '__main__':
    main()
