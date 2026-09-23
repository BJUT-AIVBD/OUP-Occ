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
    parser = argparse.ArgumentParser(description='Analyze Gaussian residual compensation effect')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_gaussian_residual_analysis_full_val')
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


def get_logits(output):
    if not isinstance(output, dict):
        return output, output, {}
    final_logits = output['final_occ_logits']
    compensated_logits = output.get('compensated_occ_logits', output.get('output_occ_logits', final_logits))
    return final_logits, compensated_logits, output


def update_region_stats(store, name, region, tuned_err, final_err, comp_err, u_total=None):
    count = int(region.sum())
    store[name]['voxel_count'] += count
    store[name]['tuned_errors'] += int(tuned_err[region].sum())
    store[name]['final_errors'] += int(final_err[region].sum())
    store[name]['compensated_errors'] += int(comp_err[region].sum())
    if u_total is not None:
        store[name]['uncertainty_sum'] += float(u_total[region].sum())


def finalize_region_stats(store):
    out = {}
    rows = []
    for name, stats in store.items():
        count = stats['voxel_count']
        tuned = stats['tuned_errors'] / count if count else float('nan')
        final = stats['final_errors'] / count if count else float('nan')
        comp = stats['compensated_errors'] / count if count else float('nan')
        item = dict(
            voxel_count=count,
            tuned_error_rate=tuned,
            final_error_rate=final,
            compensated_error_rate=comp,
            error_reduction_from_final=final - comp if count else float('nan'),
            error_reduction_from_tuned=tuned - comp if count else float('nan'),
            mean_uncertainty=stats['uncertainty_sum'] / count if count else float('nan'))
        out[name] = item
        rows.append(dict(region=name, **item))
    return out, rows


def update_gaussian_stats(acc, gaussian_dict, residual_logits, valid_mask):
    heatmap = tensor_to_numpy(gaussian_dict['gaussian_bev_heatmap']).astype(np.float32)
    high_mask = tensor_to_numpy(gaussian_dict['high_uncertainty_mask_voxel']).astype(np.float32)
    residual = tensor_to_numpy(residual_logits).astype(np.float32)
    acc['samples'] += heatmap.shape[0]
    acc['num_gaussians_sum'] += int(gaussian_dict.get('num_gaussians', 0)) * heatmap.shape[0]
    acc['heatmap_sum'] += float(heatmap.mean()) * heatmap.shape[0]
    acc['heatmap_max'] = max(acc['heatmap_max'], float(heatmap.max()))
    residual_mag = np.abs(residual).mean(axis=-1)
    acc['residual_mag_sum'] += float(residual_mag[valid_mask].mean()) if valid_mask.any() else 0.0
    acc['active_ratio_sum'] += float((high_mask[valid_mask] > 0).mean()) if valid_mask.any() else 0.0


def finalize_gaussian_stats(acc):
    samples = max(1, acc['samples'])
    return dict(
        num_gaussians=acc['num_gaussians_sum'] / samples,
        mean_heatmap_value=acc['heatmap_sum'] / samples,
        max_heatmap_value=acc['heatmap_max'],
        mean_residual_magnitude=acc['residual_mag_sum'] / samples,
        active_voxel_ratio=acc['active_ratio_sum'] / samples)


def save_panel(path, title_items, cmaps=None, vmins=None, vmaxs=None, transpose=True):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cmaps = cmaps or ['viridis'] * 4
    vmins = vmins or [None] * 4
    vmaxs = vmaxs or [None] * 4
    for ax, (title, arr), cmap, vmin, vmax in zip(axes.flat, title_items, cmaps, vmins, vmaxs):
        show = arr.T if transpose else arr
        im = ax.imshow(show, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_uncertainty_gaussian(path, u_bev, centers, heatmap, high_mask):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    items = [
        ('U_total_bev', u_bev, 'viridis'),
        ('selected gaussian centers', u_bev, 'viridis'),
        ('gaussian_bev_heatmap', heatmap, 'viridis'),
        ('high_uncertainty_mask_bev', high_mask, 'Reds'),
    ]
    for ax, (title, arr, cmap) in zip(axes.flat, items):
        im = ax.imshow(arr, origin='lower', cmap=cmap, vmin=0.0, vmax=1.0)
        if 'centers' in title and centers.size:
            ax.scatter(centers[:, 1], centers[:, 0], s=2, c='red')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(out_dir):
    with open(os.path.join(out_dir, 'README_ANALYSIS.md'), 'w') as f:
        f.write(
            '# Gaussian Residual Analysis\n\n'
            'This directory compares OUP-Occ tuned 52.13 with the Gaussian '
            'residual compensated model. Use `summary.json` for overall and '
            'region statistics, and `vis/` for per-sample residual heatmaps.\n')


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
        raise RuntimeError('CUDA is required for Gaussian residual analysis.')

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
    gaussian_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'gaussian')

    metric_tuned = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    metric_final = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    metric_comp = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    class_gt_counts = np.zeros(num_classes, dtype=np.float64)
    stat_factory = lambda: {
        'voxel_count': 0,
        'tuned_errors': 0,
        'final_errors': 0,
        'compensated_errors': 0,
        'uncertainty_sum': 0.0,
    }
    high_stats = defaultdict(stat_factory)
    visibility_stats = defaultdict(stat_factory)
    distance_stats_acc = defaultdict(stat_factory)
    gaussian_acc = defaultdict(float)
    gaussian_acc['heatmap_max'] = 0.0
    processed = 0

    for _, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            tuned_out = tuned_model(return_loss=True, return_result=True, **data_batch)
            gaussian_out = gaussian_model(return_loss=True, return_result=True, **data_batch)

        tuned_final, tuned_comp, _ = get_logits(tuned_out)
        final_logits, comp_logits, gaussian_output = get_logits(gaussian_out)
        tuned_logits = align_logits_to_bxyzc(tuned_comp, num_classes=num_classes)
        final_logits = align_logits_to_bxyzc(final_logits, num_classes=num_classes)
        comp_logits = align_logits_to_bxyzc(comp_logits, num_classes=num_classes)
        spatial_shape = tuple(comp_logits.shape[1:4])
        gt = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'voxel_semantics'), spatial_shape, 'voxel_semantics')
        mask_camera = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_camera'), spatial_shape, 'mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_lidar'), spatial_shape, 'mask_lidar').bool()
        pred_tuned = tuned_logits.softmax(-1).argmax(-1)
        pred_final = final_logits.softmax(-1).argmax(-1)
        pred_comp = comp_logits.softmax(-1).argmax(-1)
        uncertainty_dict = gaussian_output['uncertainty_dict']
        gaussian_dict = gaussian_output['gaussian_dict']
        u_total = align_tensor_to_bxyz(
            uncertainty_dict['U_total_voxel'], spatial_shape, 'U_total_voxel')
        residual_logits = align_logits_to_bxyzc(
            gaussian_dict['gaussian_residual_logits'], num_classes=num_classes)

        metas = get_img_metas(data_batch)
        batch_size = int(comp_logits.shape[0])
        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            sample_name = sample_name_from_meta(metas[b] if b < len(metas) else {}, processed)
            file_name = safe_name(sample_name)
            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            tuned_np = tensor_to_numpy(pred_tuned[b]).astype(np.int16)
            final_np = tensor_to_numpy(pred_final[b]).astype(np.int16)
            comp_np = tensor_to_numpy(pred_comp[b]).astype(np.int16)
            u_np = tensor_to_numpy(u_total[b]).astype(np.float32)
            residual_np = tensor_to_numpy(residual_logits[b]).astype(np.float32)
            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            metric_valid = mask_camera_np & valid_label
            add_official_metric(metric_tuned, tuned_np, gt_np, mask_lidar_np, mask_camera_np)
            add_official_metric(metric_final, final_np, gt_np, mask_lidar_np, mask_camera_np)
            add_official_metric(metric_comp, comp_np, gt_np, mask_lidar_np, mask_camera_np)
            for cls_idx in range(num_classes):
                class_gt_counts[cls_idx] += int(((gt_np == cls_idx) & metric_valid).sum())

            tuned_err = (tuned_np != gt_np) & valid_mask
            final_err = (final_np != gt_np) & valid_mask
            comp_err = (comp_np != gt_np) & valid_mask
            tuned_label_err = (tuned_np != gt_np) & valid_label
            final_label_err = (final_np != gt_np) & valid_label
            comp_label_err = (comp_np != gt_np) & valid_label
            corrected = final_err & ~comp_err
            degraded = ~final_err & comp_err & valid_mask

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(u_np, valid_mask, ratio)
                update_region_stats(high_stats, key, region, tuned_err, final_err, comp_err, u_np)

            regions = {
                'visible_by_both': mask_camera_np & mask_lidar_np,
                'camera_only': mask_camera_np & ~mask_lidar_np,
                'lidar_only': ~mask_camera_np & mask_lidar_np,
                'invisible_by_both': ~mask_camera_np & ~mask_lidar_np,
            }
            for name, region in regions.items():
                update_region_stats(
                    visibility_stats, name, region & valid_label,
                    tuned_label_err, final_label_err, comp_label_err, u_np)

            dist_regions = distance_masks(u_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    update_region_stats(
                        distance_stats_acc, name, region & valid_mask,
                        tuned_err, final_err, comp_err, u_np)

            high_mask_voxel = tensor_to_numpy(
                gaussian_dict['high_uncertainty_mask_voxel'][b]).astype(np.float32)
            update_gaussian_stats(
                gaussian_acc,
                {k: (v[b:b + 1] if torch.is_tensor(v) and v.shape[0] == batch_size else v)
                 for k, v in gaussian_dict.items()},
                residual_logits[b:b + 1],
                metric_valid[None, ...])

            if processed < args.vis_topk:
                heatmap = tensor_to_numpy(gaussian_dict['gaussian_bev_heatmap'][b, 0]).astype(np.float32)
                high_bev = tensor_to_numpy(gaussian_dict['high_uncertainty_mask_bev'][b, 0]).astype(np.float32)
                centers = tensor_to_numpy(gaussian_dict['gaussian_centers'][b]).astype(np.float32)
                u_bev = tensor_to_numpy(uncertainty_dict['U_total_bev_feat'][b, 0]).astype(np.float32)
                save_uncertainty_gaussian(
                    os.path.join(vis_dir, f'{file_name}_uncertainty_gaussian_heatmap.png'),
                    u_bev, centers, heatmap, high_bev)
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_residual_compensation_overlay.png'),
                    [
                        ('final error before Gaussian', bev_error(final_err)),
                        ('compensated error after Gaussian', bev_error(comp_err)),
                        ('corrected voxels', bev_error(corrected)),
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
                        ('final before Gaussian', bev_semantic(final_np, num_classes - 1)),
                        ('compensated prediction', bev_semantic(comp_np, num_classes - 1)),
                    ],
                    ['tab20', 'tab20', 'tab20', 'tab20'])
                residual_mag = np.abs(residual_np).mean(axis=-1).max(axis=2)
                corrected_small_dynamic = corrected & np.isin(gt_np, [2, 5, 6, 7, 8, 9, 10])
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_gaussian_residual_magnitude.png'),
                    [
                        ('residual magnitude BEV', residual_mag),
                        ('uncertainty map', u_np.max(axis=2)),
                        ('corrected voxels', bev_error(corrected)),
                        ('small/dynamic corrected', bev_error(corrected_small_dynamic)),
                    ],
                    ['viridis', 'viridis', 'Greens', 'Greens'])
                top10 = top_uncertainty_mask(u_np, valid_mask, 0.10)
                masked_gt = np.full_like(gt_np, num_classes - 1)
                masked_final = np.full_like(final_np, num_classes - 1)
                masked_comp = np.full_like(comp_np, num_classes - 1)
                masked_gt[top10] = gt_np[top10]
                masked_final[top10] = final_np[top10]
                masked_comp[top10] = comp_np[top10]
                save_panel(
                    os.path.join(vis_dir, f'{file_name}_top_uncertainty_compensation.png'),
                    [
                        ('top10 GT', bev_semantic(masked_gt, num_classes - 1)),
                        ('top10 final', bev_semantic(masked_final, num_classes - 1)),
                        ('top10 compensated', bev_semantic(masked_comp, num_classes - 1)),
                        ('top10 corrected', bev_error(corrected & top10)),
                    ],
                    ['tab20', 'tab20', 'tab20', 'Greens'])

            processed += 1
            logger.info('Processed %s', sample_name)

    tuned_miou, tuned_iou = miou_from_metric(metric_tuned)
    final_miou, final_iou = miou_from_metric(metric_final)
    comp_miou, comp_iou = miou_from_metric(metric_comp)
    high_summary, high_rows = finalize_region_stats(high_stats)
    visibility_summary, visibility_rows = finalize_region_stats(visibility_stats)
    distance_summary, distance_rows = finalize_region_stats(distance_stats_acc)
    gaussian_stats = finalize_gaussian_stats(gaussian_acc)

    per_class_rows = []
    for cls_idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=cls_idx,
            class_name=class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            num_gt_voxels=int(class_gt_counts[cls_idx]),
            tuned_iou=float(tuned_iou[cls_idx]),
            final_before_gaussian_iou=float(final_iou[cls_idx]),
            compensated_iou=float(comp_iou[cls_idx]),
            compensated_minus_tuned=float(comp_iou[cls_idx] - tuned_iou[cls_idx]),
            compensated_minus_final=float(comp_iou[cls_idx] - final_iou[cls_idx])))

    summary = dict(
        overall=dict(
            tuned_mIoU=tuned_miou,
            final_before_gaussian_mIoU=final_miou,
            compensated_mIoU=comp_miou,
            compensated_minus_tuned=comp_miou - tuned_miou,
            compensated_minus_final=comp_miou - final_miou),
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
                and not math.isnan(row['compensated_minus_tuned'])
            ],
            key=lambda x: x['compensated_minus_tuned'],
            reverse=True)[:10],
        gaussian_statistics=gaussian_stats)
    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(os.path.join(args.out_dir, 'high_uncertainty_improvement.csv'), high_rows)
    write_csv(os.path.join(args.out_dir, 'visibility_region_improvement.csv'), visibility_rows)
    write_csv(os.path.join(args.out_dir, 'distance_region_improvement.csv'), distance_rows)
    write_csv(os.path.join(args.out_dir, 'gaussian_statistics.csv'), [gaussian_stats])
    write_readme(os.path.join(args.out_dir))
    logger.info('Saved Gaussian residual analysis to %s', args.out_dir)
    print('\n==== Gaussian Residual Effect Summary ====')
    print(f'   - tuned mIoU: {tuned_miou:.4f}')
    print(f'   - final before Gaussian mIoU: {final_miou:.4f}')
    print(f'   - compensated mIoU: {comp_miou:.4f}')
    print(f'   - compensated - tuned: {comp_miou - tuned_miou:.4f}')
    print(f'   - compensated - final: {comp_miou - final_miou:.4f}')


if __name__ == '__main__':
    main()
