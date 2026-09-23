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
    parser = argparse.ArgumentParser(description='Analyze UCRF reliability fusion effect')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_ucrf_analysis_full_val')
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


def setup_logger(out_dir):
    logs_dir = os.path.join(out_dir, 'logs')
    mmcv.mkdir_or_exist(logs_dir)
    import logging
    logger = logging.getLogger('reliability_fusion_effect')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'analysis.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


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


def update_region_stats(store, name, region, tuned_err, ucrf_err, r_cam=None, r_lidar=None, d_abs=None):
    count = int(region.sum())
    store[name]['voxel_count'] += count
    store[name]['tuned_errors'] += int(tuned_err[region].sum())
    store[name]['ucrf_errors'] += int(ucrf_err[region].sum())
    if r_cam is not None:
        store[name]['r_cam_sum'] += float(r_cam[region].sum())
    if r_lidar is not None:
        store[name]['r_lidar_sum'] += float(r_lidar[region].sum())
    if d_abs is not None:
        store[name]['d_abs_sum'] += float(d_abs[region].sum())


def finalize_region_stats(store):
    out = {}
    rows = []
    for name, stats in store.items():
        count = stats['voxel_count']
        tuned = stats['tuned_errors'] / count if count else float('nan')
        ucrf = stats['ucrf_errors'] / count if count else float('nan')
        item = dict(
            voxel_count=count,
            tuned_error_rate=tuned,
            ucrf_error_rate=ucrf,
            error_reduction=tuned - ucrf if count else float('nan'),
            mean_R_cam=stats['r_cam_sum'] / count if count else float('nan'),
            mean_R_lidar=stats['r_lidar_sum'] / count if count else float('nan'),
            mean_discrepancy=stats['d_abs_sum'] / count if count else float('nan'))
        out[name] = item
        rows.append(dict(region=name, **item))
    return out, rows


def project_bev_to_voxel(bev, shape):
    xdim, ydim, zdim = shape
    if bev.shape != (ydim, xdim):
        bev_t = torch.from_numpy(bev).float().view(1, 1, *bev.shape)
        bev = torch.nn.functional.interpolate(bev_t, size=(ydim, xdim), mode='nearest')[0, 0].numpy()
    xy = bev.T
    return np.repeat(xy[:, :, None], zdim, axis=2)


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


def update_reliability_acc(acc, r_cam, r_lidar, d_abs, residual_norm, regions):
    acc['count'] += 1
    acc['mean_R_cam'] += float(r_cam.mean())
    acc['mean_R_lidar'] += float(r_lidar.mean())
    acc['mean_discrepancy'] += float(d_abs.mean())
    acc['mean_residual_norm'] += float(residual_norm)
    for name, region in regions.items():
        bev_region = region.max(axis=2).T
        if bev_region.shape != r_cam.shape:
            bev_region = project_bev_to_voxel(bev_region, region.shape).max(axis=2).T
        count = int(bev_region.sum())
        if count:
            acc[f'R_cam_{name}'] += float(r_cam[bev_region].mean())
            acc[f'R_lidar_{name}'] += float(r_lidar[bev_region].mean())
            acc[f'count_{name}'] += 1


def finalize_reliability_acc(acc):
    count = max(1, acc['count'])
    out = dict(
        mean_R_cam=acc['mean_R_cam'] / count,
        mean_R_lidar=acc['mean_R_lidar'] / count,
        mean_discrepancy=acc['mean_discrepancy'] / count,
        mean_residual_norm=acc['mean_residual_norm'] / count)
    for name in ['visible_by_both', 'camera_only', 'lidar_only', 'invisible_by_both']:
        c = max(1, acc.get(f'count_{name}', 0))
        out[f'mean_R_cam_{name}'] = acc.get(f'R_cam_{name}', 0.0) / c
        out[f'mean_R_lidar_{name}'] = acc.get(f'R_lidar_{name}', 0.0) / c
    return out


def write_readme(out_dir):
    with open(os.path.join(out_dir, 'README_ANALYSIS.md'), 'w') as f:
        f.write(
            '# UCRF Analysis\n\n'
            'This directory compares OUP-Occ tuned 52.13 with OUP-Occ + UCRF. '
            'Reliability maps show camera/LiDAR softmax weights and discrepancy.\n')


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
        raise RuntimeError('CUDA is required for UCRF analysis.')

    dataset_cfg = patch_analysis_pipeline(cfg.data[args.split])
    dataset = build_dataset(dataset_cfg)
    full_val_size = len(dataset)
    sample_limit = None if args.full_val or args.max_samples is None or args.max_samples < 0 else args.max_samples
    if sample_limit is not None and sample_limit >= full_val_size:
        sample_limit = None
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=args.workers_per_gpu,
        dist=False, shuffle=False)

    num_classes = get_num_classes(cfg)
    ignore_index = get_ignore_index(cfg, args)
    class_names = Metric_mIoU(num_classes=num_classes).class_names[:num_classes]
    tuned_model = build_eval_model(baseline_cfg, args.baseline_ckpt, args.gpu_id, dataset, logger, 'tuned')
    ucrf_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'ucrf')
    metric_tuned = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    metric_ucrf = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)

    class_gt_counts = np.zeros(num_classes, dtype=np.float64)
    stat_factory = lambda: {
        'voxel_count': 0, 'tuned_errors': 0, 'ucrf_errors': 0,
        'r_cam_sum': 0.0, 'r_lidar_sum': 0.0, 'd_abs_sum': 0.0
    }
    high_stats = defaultdict(stat_factory)
    visibility_stats = defaultdict(stat_factory)
    distance_stats_acc = defaultdict(stat_factory)
    reliability_acc = defaultdict(float)
    processed = 0

    for _, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            tuned_out = tuned_model(return_loss=True, return_result=True, **data_batch)
            ucrf_out = ucrf_model(return_loss=True, return_result=True, **data_batch)

        tuned_logits = align_logits_to_bxyzc(
            tuned_out.get('output_occ_logits', tuned_out['final_occ_logits']), num_classes=num_classes)
        ucrf_logits = align_logits_to_bxyzc(
            ucrf_out.get('output_occ_logits', ucrf_out['final_occ_logits']), num_classes=num_classes)
        spatial_shape = tuple(ucrf_logits.shape[1:4])
        gt = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'voxel_semantics'), spatial_shape, 'voxel_semantics')
        mask_camera = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_camera'), spatial_shape, 'mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_lidar'), spatial_shape, 'mask_lidar').bool()
        pred_tuned = tuned_logits.softmax(-1).argmax(-1)
        pred_ucrf = ucrf_logits.softmax(-1).argmax(-1)
        unc = ucrf_out['uncertainty_dict']
        rel = ucrf_out.get('reliability_dict', {})
        u_bev = unc.get('U_total_bev_feat', unc.get('U_refinement_bev_feat'))
        gate_map = unc.get('gate_map', u_bev)
        metas = get_img_metas(data_batch)
        batch_size = int(ucrf_logits.shape[0])

        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            sample_name = sample_name_from_meta(metas[b] if b < len(metas) else {}, processed)
            file_name = safe_name(sample_name)
            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            tuned_np = tensor_to_numpy(pred_tuned[b]).astype(np.int16)
            ucrf_np = tensor_to_numpy(pred_ucrf[b]).astype(np.int16)
            u_np = tensor_to_numpy(u_bev[b, 0]).astype(np.float32)
            gate_np = tensor_to_numpy(gate_map[b]).astype(np.float32)
            if gate_np.ndim == 3:
                gate_np = gate_np.mean(axis=0)

            r_cam = tensor_to_numpy(rel['R_cam'][b, 0]).astype(np.float32)
            r_lidar = tensor_to_numpy(rel['R_lidar'][b, 0]).astype(np.float32)
            d_abs = tensor_to_numpy(rel['D_abs'][b, 0]).astype(np.float32)
            residual = tensor_to_numpy(rel['reliability_residual'][b]).astype(np.float32)
            residual_norm = float(np.sqrt(np.mean(residual ** 2)))
            r_cam_voxel = project_bev_to_voxel(r_cam, gt_np.shape)
            r_lidar_voxel = project_bev_to_voxel(r_lidar, gt_np.shape)
            d_abs_voxel = project_bev_to_voxel(d_abs, gt_np.shape)

            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            add_official_metric(metric_tuned, tuned_np, gt_np, mask_lidar_np, mask_camera_np)
            add_official_metric(metric_ucrf, ucrf_np, gt_np, mask_lidar_np, mask_camera_np)
            metric_valid = mask_camera_np & valid_label
            for cls_idx in range(num_classes):
                class_gt_counts[cls_idx] += int(((gt_np == cls_idx) & metric_valid).sum())

            tuned_err = (tuned_np != gt_np) & valid_mask
            ucrf_err = (ucrf_np != gt_np) & valid_mask
            tuned_label_err = (tuned_np != gt_np) & valid_label
            ucrf_label_err = (ucrf_np != gt_np) & valid_label
            corrected = tuned_err & ~ucrf_err
            degraded = ~tuned_err & ucrf_err & valid_mask
            u_voxel = project_bev_to_voxel(u_np, gt_np.shape)

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(u_voxel, valid_mask, ratio)
                update_region_stats(high_stats, key, region, tuned_err, ucrf_err, r_cam_voxel, r_lidar_voxel, d_abs_voxel)

            regions = {
                'visible_by_both': mask_camera_np & mask_lidar_np,
                'camera_only': mask_camera_np & ~mask_lidar_np,
                'lidar_only': ~mask_camera_np & mask_lidar_np,
                'invisible_by_both': ~mask_camera_np & ~mask_lidar_np,
            }
            for name, region in regions.items():
                update_region_stats(
                    visibility_stats, name, region & valid_label,
                    tuned_label_err, ucrf_label_err, r_cam_voxel, r_lidar_voxel, d_abs_voxel)

            dist_regions = distance_masks(gt_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    update_region_stats(
                        distance_stats_acc, name, region & valid_mask,
                        tuned_err, ucrf_err, r_cam_voxel, r_lidar_voxel, d_abs_voxel)

            update_reliability_acc(reliability_acc, r_cam, r_lidar, d_abs, residual_norm, regions)

            if processed < args.vis_topk:
                visibility_map = (
                    regions['visible_by_both'].astype(float)
                    + 2 * regions['camera_only'].astype(float)
                    + 3 * regions['lidar_only'].astype(float)
                    + 4 * regions['invisible_by_both'].astype(float))
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_reliability_maps.png'),
                    [
                        ('R_cam', r_cam),
                        ('R_lidar', r_lidar),
                        ('cross-modal discrepancy D_abs', d_abs),
                        ('visibility region map', visibility_map.max(axis=2)),
                    ],
                    ['viridis', 'viridis', 'magma', 'tab20'],
                    [0.0, 0.0, 0.0, None],
                    [1.0, 1.0, 1.0, None])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_reliability_vs_uncertainty.png'),
                    [
                        ('U_total_bev', u_np),
                        ('R_cam', r_cam),
                        ('R_lidar', r_lidar),
                        ('refinement gate', gate_np),
                    ],
                    ['viridis', 'viridis', 'viridis', 'viridis'],
                    [0.0] * 4,
                    [1.0] * 4)
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_tuned_vs_ucrf_error_overlay.png'),
                    [
                        ('tuned error', bev_error(tuned_err)),
                        ('UCRF error', bev_error(ucrf_err)),
                        ('improved voxels', bev_error(corrected)),
                        ('degraded voxels', bev_error(degraded)),
                    ],
                    ['Reds', 'Reds', 'Greens', 'Reds'],
                    [0.0] * 4,
                    [1.0] * 4)
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_semantic_comparison.png'),
                    [
                        ('GT BEV semantic', bev_semantic(gt_np, num_classes - 1)),
                        ('OUP tuned prediction', bev_semantic(tuned_np, num_classes - 1)),
                        ('UCRF prediction', bev_semantic(ucrf_np, num_classes - 1)),
                        ('improved regions', bev_error(corrected)),
                    ],
                    ['tab20', 'tab20', 'tab20', 'Greens'])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_visibility_region_comparison.png'),
                    [
                        ('visibility region map', visibility_map.max(axis=2)),
                        ('tuned error', bev_error(tuned_err)),
                        ('UCRF error', bev_error(ucrf_err)),
                        ('improved region', bev_error(corrected)),
                    ],
                    ['tab20', 'Reds', 'Reds', 'Greens'])
            processed += 1
            logger.info('Processed %s', sample_name)

    tuned_miou, tuned_iou = miou_from_metric(metric_tuned)
    ucrf_miou, ucrf_iou = miou_from_metric(metric_ucrf)
    high_summary, high_rows = finalize_region_stats(high_stats)
    visibility_summary, visibility_rows = finalize_region_stats(visibility_stats)
    distance_summary, distance_rows = finalize_region_stats(distance_stats_acc)
    reliability_stats = finalize_reliability_acc(reliability_acc)

    per_class_rows = []
    for cls_idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=cls_idx,
            class_name=class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            num_gt_voxels=int(class_gt_counts[cls_idx]),
            tuned_iou=float(tuned_iou[cls_idx]),
            ucrf_iou=float(ucrf_iou[cls_idx]),
            ucrf_minus_tuned=float(ucrf_iou[cls_idx] - tuned_iou[cls_idx])))

    summary = dict(
        overall=dict(
            tuned_mIoU=tuned_miou,
            ucrf_mIoU=ucrf_miou,
            ucrf_minus_tuned=ucrf_miou - tuned_miou),
        evaluation_protocol=dict(
            num_samples=processed,
            full_val_size=full_val_size,
            is_full_val=(processed == full_val_size),
            metric_style='official' if args.official_style_metric else 'custom',
            eval_mask=args.eval_mask,
            include_free_in_mIoU=False,
            class_names=class_names),
        high_uncertainty_regions=high_summary,
        visibility_regions=visibility_summary,
        distance_regions=distance_summary,
        per_class_top_improvements=sorted(
            [
                row for row in per_class_rows
                if row['class_index'] != num_classes - 1
                and not math.isnan(row['ucrf_minus_tuned'])
            ],
            key=lambda x: x['ucrf_minus_tuned'],
            reverse=True)[:10],
        reliability_statistics=reliability_stats)
    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(os.path.join(args.out_dir, 'high_uncertainty_improvement.csv'), high_rows)
    write_csv(os.path.join(args.out_dir, 'visibility_region_improvement.csv'), visibility_rows)
    write_csv(os.path.join(args.out_dir, 'distance_region_improvement.csv'), distance_rows)
    write_csv(os.path.join(args.out_dir, 'reliability_statistics.csv'), [reliability_stats])
    write_readme(args.out_dir)
    logger.info('Saved UCRF analysis to %s', args.out_dir)
    print('\n==== UCRF Analysis Summary ====')
    print(f'   - tuned mIoU: {tuned_miou:.4f}')
    print(f'   - UCRF mIoU: {ucrf_miou:.4f}')
    print(f'   - UCRF - tuned: {ucrf_miou - tuned_miou:.4f}')


if __name__ == '__main__':
    main()
