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


CLASS_NAMES = Metric_mIoU(num_classes=18).class_names
GROUPS = {
    'small_object': ['bicycle', 'motorcycle', 'pedestrian', 'traffic_cone'],
    'dynamic_object': [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'motorcycle', 'bicycle', 'pedestrian'
    ],
    'background': ['driveable_surface', 'sidewalk', 'terrain', 'manmade', 'vegetation'],
}
DEFAULT_CLASS_DIFFICULTY = {
    2: 1.0,
    3: 0.4,
    4: 0.3,
    5: 0.8,
    6: 1.0,
    7: 0.8,
    8: 1.0,
    9: 0.6,
    10: 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze UHAS hard-region supervision')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_uhas_analysis_full_val')
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
        for key in row.keys():
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


def get_final_logits(output):
    if isinstance(output, dict):
        return output['final_occ_logits'], output.get('uncertainty_dict', {})
    return output, {}


def get_hard_cfg(cfg):
    ref = cfg.model.uncertainty_refinement
    hard = ref.get('hard_region_loss', {})
    return dict(
        lambda_hard=float(ref.get('lambda_hard', 0.0)),
        lambda_u=float(hard.get('lambda_u', 1.0)),
        lambda_partial=float(hard.get('lambda_partial', 0.3)),
        lambda_invisible=float(hard.get('lambda_invisible', 0.0)),
        lambda_far=float(hard.get('lambda_far', 0.3)),
        lambda_class=float(hard.get('lambda_class', 0.3)),
        weight_max=float(hard.get('weight_max', 3.0)),
        normalize_weight=bool(hard.get('normalize_weight', True)),
        class_difficulty=hard.get('class_difficulty', DEFAULT_CLASS_DIFFICULTY))


def class_component(gt, num_classes, class_difficulty):
    values = np.zeros(num_classes, dtype=np.float32)
    for key, value in class_difficulty.items():
        idx = int(key)
        if 0 <= idx < num_classes:
            values[idx] = float(value)
    safe_gt = np.clip(gt, 0, num_classes - 1)
    return values[safe_gt]


def distance_component(shape, cfg, logger):
    dist_regions = distance_masks(shape, cfg, logger)
    comp = np.zeros(shape, dtype=np.float32)
    if dist_regions is None:
        return comp, {}
    comp[dist_regions['15-30m']] = 0.3
    comp[dist_regions['30-45m']] = 0.6
    comp[dist_regions['45m+']] = 1.0
    return comp, dist_regions


def hard_weight_map(gt, u_total, mask_camera, mask_lidar, valid_mask, cfg, num_classes, logger):
    hard_cfg = get_hard_cfg(cfg)
    partial = ((mask_camera & ~mask_lidar) | (~mask_camera & mask_lidar)).astype(np.float32)
    invisible = (~mask_camera & ~mask_lidar).astype(np.float32)
    far, dist_regions = distance_component(gt.shape, cfg, logger)
    cls = class_component(gt, num_classes, hard_cfg['class_difficulty'])
    weight = (
        1.0
        + hard_cfg['lambda_u'] * u_total
        + hard_cfg['lambda_partial'] * partial
        + hard_cfg['lambda_invisible'] * invisible
        + hard_cfg['lambda_far'] * far
        + hard_cfg['lambda_class'] * cls)
    weight = np.clip(weight, 1.0, hard_cfg['weight_max'])
    if hard_cfg['normalize_weight'] and valid_mask.any():
        weight = weight / max(float(weight[valid_mask].mean()), 1e-6)
    components = dict(
        uncertainty=hard_cfg['lambda_u'] * u_total,
        partial_visibility=hard_cfg['lambda_partial'] * partial,
        invisible=hard_cfg['lambda_invisible'] * invisible,
        distance=hard_cfg['lambda_far'] * far,
        class_difficulty=hard_cfg['lambda_class'] * cls)
    return weight.astype(np.float32), components, dist_regions


def update_compare_stats(store, name, region, tuned_err, uhas_err, u_total=None, weight=None):
    count = int(region.sum())
    store[name]['voxel_count'] += count
    store[name]['tuned_errors'] += int(tuned_err[region].sum())
    store[name]['uhas_errors'] += int(uhas_err[region].sum())
    if u_total is not None:
        store[name]['uncertainty_sum'] += float(u_total[region].sum())
    if weight is not None:
        store[name]['weight_sum'] += float(weight[region].sum())


def finalize_compare_stats(store):
    rows = []
    out = {}
    for name, stats in store.items():
        count = stats['voxel_count']
        tuned = stats['tuned_errors'] / count if count else float('nan')
        uhas = stats['uhas_errors'] / count if count else float('nan')
        item = dict(
            voxel_count=count,
            tuned_error_rate=tuned,
            uhas_error_rate=uhas,
            error_reduction=tuned - uhas if count else float('nan'),
            mean_uncertainty=stats['uncertainty_sum'] / count if count else float('nan'),
            mean_weight=stats['weight_sum'] / count if count else float('nan'))
        out[name] = item
        rows.append(dict(region=name, **item))
    return out, rows


def update_weight_acc(acc, valid, weight, components):
    count = int(valid.sum())
    if count == 0:
        return
    acc['count'] += count
    valid_weight = weight[valid]
    acc['weight_sum'] += float(valid_weight.sum())
    acc['weight_min'] = min(acc['weight_min'], float(valid_weight.min()))
    acc['weight_max'] = max(acc['weight_max'], float(valid_weight.max()))
    acc['gt_1_5'] += int((valid_weight > 1.5).sum())
    acc['gt_2_0'] += int((valid_weight > 2.0).sum())
    for name, comp in components.items():
        acc[f'{name}_sum'] += float(comp[valid].sum())


def finalize_weight_acc(acc):
    count = acc['count']
    if count == 0:
        return {}
    return dict(
        mean_weight=acc['weight_sum'] / count,
        max_weight=acc['weight_max'],
        min_weight=acc['weight_min'],
        mean_uncertainty_weight=acc['uncertainty_sum'] / count,
        mean_partial_visibility_weight=acc['partial_visibility_sum'] / count,
        mean_invisible_weight=acc['invisible_sum'] / count,
        mean_distance_weight=acc['distance_sum'] / count,
        mean_class_weight=acc['class_difficulty_sum'] / count,
        ratio_weight_gt_1_5=acc['gt_1_5'] / count,
        ratio_weight_gt_2_0=acc['gt_2_0'] / count)


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


def write_readme(out_dir):
    with open(os.path.join(out_dir, 'README_ANALYSIS.md'), 'w') as f:
        f.write(
            '# UHAS Analysis\n\n'
            'This directory compares the tuned OUP-Occ refinement model against '
            'the UHAS model. Full-val results are directly comparable to official '
            'validation when `evaluation_protocol.is_full_val=true` and '
            '`metric_style=official`.\n')


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
        raise RuntimeError('CUDA is required for UHAS analysis.')

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
    uhas_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'uhas')
    metric_tuned = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    metric_uhas = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)

    stat_factory = lambda: {
        'voxel_count': 0,
        'tuned_errors': 0,
        'uhas_errors': 0,
        'uncertainty_sum': 0.0,
        'weight_sum': 0.0
    }
    hard_stats = defaultdict(stat_factory)
    visibility_stats = defaultdict(stat_factory)
    distance_stats_acc = defaultdict(stat_factory)
    weight_acc = defaultdict(float)
    weight_acc['weight_min'] = float('inf')
    weight_acc['weight_max'] = float('-inf')

    processed = 0
    for batch_idx, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            tuned_out = tuned_model(return_loss=True, return_result=True, **data_batch)
            uhas_out = uhas_model(return_loss=True, return_result=True, **data_batch)

        tuned_logits, _ = get_final_logits(tuned_out)
        uhas_logits, uhas_unc = get_final_logits(uhas_out)
        tuned_logits = align_logits_to_bxyzc(tuned_logits, num_classes=num_classes)
        uhas_logits = align_logits_to_bxyzc(uhas_logits, num_classes=num_classes)
        spatial_shape = tuple(uhas_logits.shape[1:4])

        gt = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'voxel_semantics'), spatial_shape, 'voxel_semantics')
        mask_camera = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_camera'), spatial_shape, 'mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_lidar'), spatial_shape, 'mask_lidar').bool()
        pred_tuned = tuned_logits.softmax(-1).argmax(-1)
        pred_uhas = uhas_logits.softmax(-1).argmax(-1)
        u_total = uhas_unc.get('U_total_voxel')
        if u_total is None:
            raise KeyError('UHAS model output does not contain uncertainty_dict[U_total_voxel].')
        u_total = align_tensor_to_bxyz(u_total, spatial_shape, 'U_total_voxel')
        gate = uhas_unc.get('gate_map', None)

        metas = get_img_metas(data_batch)
        batch_size = int(uhas_logits.shape[0])
        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            sample_name = sample_name_from_meta(metas[b] if b < len(metas) else {}, processed)
            file_name = safe_name(sample_name)
            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            tuned_np = tensor_to_numpy(pred_tuned[b]).astype(np.int16)
            uhas_np = tensor_to_numpy(pred_uhas[b]).astype(np.int16)
            u_np = tensor_to_numpy(u_total[b]).astype(np.float32)
            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            metric_valid = mask_camera_np & valid_label
            metric_tuned.add_batch(tuned_np.copy(), gt_np.copy(), mask_lidar_np.copy(), mask_camera_np.copy())
            metric_uhas.add_batch(uhas_np.copy(), gt_np.copy(), mask_lidar_np.copy(), mask_camera_np.copy())

            weight, components, dist_regions = hard_weight_map(
                gt_np, u_np, mask_camera_np, mask_lidar_np, metric_valid, cfg, num_classes, logger)
            update_weight_acc(weight_acc, metric_valid, weight, components)
            tuned_err = (tuned_np != gt_np) & valid_mask
            uhas_err = (uhas_np != gt_np) & valid_mask
            tuned_label_err = (tuned_np != gt_np) & valid_label
            uhas_label_err = (uhas_np != gt_np) & valid_label

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(u_np, valid_mask, ratio)
                update_compare_stats(hard_stats, key, region, tuned_err, uhas_err, u_np, weight)

            regions = {
                'visible_by_both': mask_camera_np & mask_lidar_np,
                'camera_only': mask_camera_np & ~mask_lidar_np,
                'lidar_only': ~mask_camera_np & mask_lidar_np,
                'invisible_by_both': ~mask_camera_np & ~mask_lidar_np,
            }
            for name, region in regions.items():
                update_compare_stats(
                    visibility_stats, name, region & valid_label,
                    tuned_label_err, uhas_label_err, u_np, weight)

            if dist_regions:
                for name, region in dist_regions.items():
                    update_compare_stats(
                        distance_stats_acc, name, region & valid_mask, tuned_err, uhas_err, u_np, weight)

            if processed < args.vis_topk:
                improved = tuned_err & ~uhas_err
                degraded = ~tuned_err & uhas_err & valid_mask
                gate_np = np.zeros_like(u_np[:, :, 0])
                if gate is not None:
                    gate_np = tensor_to_numpy(gate[b]).astype(np.float32)
                    if gate_np.ndim == 3:
                        gate_np = gate_np.mean(axis=0)
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_hard_weight_map.png'),
                    [
                        ('U_total_bev', u_np.max(axis=2)),
                        ('hard supervision weight BEV', weight.max(axis=2)),
                        ('gate map', gate_np),
                        ('UHAS final error', bev_error(uhas_err)),
                    ],
                    ['viridis', 'viridis', 'viridis', 'Reds'],
                    [0.0, None, 0.0, 0.0],
                    [1.0, None, 1.0, 1.0])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_tuned_vs_uhas_error_overlay.png'),
                    [
                        ('tuned error', bev_error(tuned_err)),
                        ('UHAS error', bev_error(uhas_err)),
                        ('improved voxels', bev_error(improved)),
                        ('degraded voxels', bev_error(degraded)),
                    ],
                    ['Reds', 'Reds', 'Greens', 'Reds'],
                    [0.0] * 4,
                    [1.0] * 4)
                visibility_map = (
                    regions['visible_by_both'].astype(float)
                    + 2 * regions['camera_only'].astype(float)
                    + 3 * regions['lidar_only'].astype(float)
                    + 4 * regions['invisible_by_both'].astype(float))
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_distance_visibility_weight.png'),
                    [
                        ('distance component', components['distance'].max(axis=2)),
                        ('visibility region', visibility_map.max(axis=2)),
                        ('hard weight', weight.max(axis=2)),
                        ('corrected region', bev_error(improved)),
                    ],
                    ['viridis', 'tab20', 'viridis', 'Greens'])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_semantic_comparison.png'),
                    [
                        ('GT BEV semantic', bev_semantic(gt_np, num_classes - 1)),
                        ('tuned prediction', bev_semantic(tuned_np, num_classes - 1)),
                        ('UHAS prediction', bev_semantic(uhas_np, num_classes - 1)),
                        ('improved regions', bev_error(improved)),
                    ],
                    ['tab20', 'tab20', 'tab20', 'Greens'])

            processed += 1
            logger.info('Processed %s', sample_name)

    tuned_miou, tuned_iou = miou_from_metric(metric_tuned)
    uhas_miou, uhas_iou = miou_from_metric(metric_uhas)
    hard_summary, hard_rows = finalize_compare_stats(hard_stats)
    visibility_summary, visibility_rows = finalize_compare_stats(visibility_stats)
    distance_summary, distance_rows = finalize_compare_stats(distance_stats_acc)
    weight_summary = finalize_weight_acc(weight_acc)

    per_class_rows = []
    for idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=idx,
            class_name=class_names[idx] if idx < len(class_names) else str(idx),
            tuned_iou=float(tuned_iou[idx]),
            uhas_iou=float(uhas_iou[idx]),
            uhas_minus_tuned=float(uhas_iou[idx] - tuned_iou[idx])))

    group_rows = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for group_name, names in GROUPS.items():
        indices = [class_to_idx[name] for name in names if name in class_to_idx]
        tuned_group = float(np.nanmean(tuned_iou[indices])) if indices else float('nan')
        uhas_group = float(np.nanmean(uhas_iou[indices])) if indices else float('nan')
        group_rows.append(dict(
            group=group_name,
            classes=','.join(names),
            tuned_mIoU=tuned_group,
            uhas_mIoU=uhas_group,
            uhas_minus_tuned=uhas_group - tuned_group))

    summary = dict(
        overall=dict(
            tuned_mIoU=tuned_miou,
            uhas_mIoU=uhas_miou,
            uhas_minus_tuned=uhas_miou - tuned_miou),
        evaluation_protocol=dict(
            num_samples=processed,
            full_val_size=full_val_size,
            is_full_val=(processed == full_val_size),
            metric_style='official' if args.official_style_metric else 'custom',
            eval_mask=args.eval_mask,
            include_free_in_mIoU=False,
            class_names=class_names),
        hard_uncertainty_regions=hard_summary,
        visibility_regions=visibility_summary,
        distance_regions=distance_summary,
        class_groups={row['group']: row for row in group_rows},
        weight_statistics=weight_summary)

    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(os.path.join(args.out_dir, 'group_iou.csv'), group_rows)
    write_csv(os.path.join(args.out_dir, 'hard_region_improvement.csv'), hard_rows)
    write_csv(os.path.join(args.out_dir, 'visibility_region_improvement.csv'), visibility_rows)
    write_csv(os.path.join(args.out_dir, 'distance_region_improvement.csv'), distance_rows)
    write_csv(os.path.join(args.out_dir, 'weight_statistics.csv'), [weight_summary])
    write_readme(args.out_dir)
    logger.info('Saved UHAS analysis to %s', args.out_dir)
    print('\n==== UHAS Analysis Summary ====')
    print(f'   - tuned mIoU: {tuned_miou:.4f}')
    print(f'   - UHAS mIoU: {uhas_miou:.4f}')
    print(f'   - UHAS - tuned: {uhas_miou - tuned_miou:.4f}')


if __name__ == '__main__':
    main()
