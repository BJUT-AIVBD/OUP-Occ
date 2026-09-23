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
import torch.nn.functional as F
from mmcv import Config, DictAction

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

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
    bev_error,
    bev_semantic,
    build_eval_model,
    hist_info,
    metric_mask_name,
    miou_from_hist,
    per_class_iou,
    safe_name,
    safe_rate,
    top_uncertainty_mask,
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze UCRF + UDCA effect')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_ucrf_udca_analysis_full_val')
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
    logger = logging.getLogger('ucrf_udca_effect')
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


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(values.mean())


def write_csv(path, rows):
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
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


def extract_logits(output, prefer_final=True):
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, dict):
        keys = ['output_occ_logits', 'compensated_occ_logits', 'final_occ_logits']
        if not prefer_final:
            keys = ['coarse_occ_logits'] + keys
        for key in keys:
            if key in output:
                return output[key]
        raise KeyError('No occupancy logits found in model output dict.')
    return output


def map_tensor(output, key, fallback=None):
    if not isinstance(output, dict):
        return fallback
    return output.get(key, fallback)


def to_numpy_2d(tensor, index=0):
    if tensor is None:
        return None
    while isinstance(tensor, (list, tuple)) and len(tensor) == 1:
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().float().cpu()
        if tensor.dim() == 4:
            tensor = tensor[index, 0]
        elif tensor.dim() == 3:
            tensor = tensor[index]
        return tensor.numpy()
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        return arr[index, 0]
    if arr.ndim == 3:
        return arr[index]
    return arr


def resize_feature_bev_to_xy(array, xy_shape):
    if array is None:
        return np.full(xy_shape, np.nan, dtype=np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32).view(1, 1, array.shape[0], array.shape[1])
    tensor = F.interpolate(tensor, size=(xy_shape[1], xy_shape[0]), mode='bilinear', align_corners=False)
    # Feature maps are [H=y, W=x]; labels are [X, Y, Z].
    return tensor[0, 0].numpy().T


def feature_mask_to_xy(mask, xy_shape):
    return resize_feature_bev_to_xy(mask.astype(np.float32), xy_shape) > 0.5


def scatter_query_map(indices, values, hw, xy_shape):
    out = np.zeros(hw, dtype=np.float32)
    if indices is None or values is None:
        return resize_feature_bev_to_xy(out, xy_shape)
    indices = np.asarray(indices).reshape(-1)
    values = np.asarray(values).reshape(-1)
    h, w = hw
    valid = (indices >= 0) & (indices < h * w)
    out.reshape(-1)[indices[valid]] = values[valid]
    return resize_feature_bev_to_xy(out, xy_shape)


def visibility_region_map(mask_camera, mask_lidar):
    cam = mask_camera.max(axis=2).astype(np.int16)
    lidar = mask_lidar.max(axis=2).astype(np.int16)
    return cam + 2 * lidar


def plot_grid(items, path, sample_name, cmap='viridis'):
    cols = min(3, len(items))
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, items):
        title, arr, item_cmap, vmin, vmax = item
        im = ax.imshow(arr.T, origin='lower', cmap=item_cmap or cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(items):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_udca_offsets(sample_name, u_bev_xy, selected_xy, offsets_cam, offsets_lidar, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(u_bev_xy.T, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
    axes[0].imshow(np.ma.masked_where(~selected_xy.T, selected_xy.T), origin='lower', cmap='Reds', alpha=0.55)
    axes[0].set_title(f'{sample_name} selected queries')
    for ax, offsets, title, color in [
            (axes[1], offsets_cam, 'camera offsets', 'cyan'),
            (axes[2], offsets_lidar, 'lidar offsets', 'yellow')]:
        ax.imshow(u_bev_xy.T, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
        ys, xs = np.nonzero(selected_xy.T)
        if offsets is not None and xs.size > 0:
            mean_offsets = np.asarray(offsets).mean(axis=1)
            stride = max(1, xs.size // 200)
            ax.quiver(
                xs[::stride],
                ys[::stride],
                mean_offsets[:xs.size:stride, 1],
                mean_offsets[:xs.size:stride, 0],
                color=color,
                angles='xy',
                scale_units='xy',
                scale=1.0)
        ax.set_title(f'{sample_name} {title}')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def update_pair_stats(store, name, region, tuned_err, ours_err):
    count = int(region.sum())
    store[name]['count'] += count
    store[name]['tuned_errors'] += int(tuned_err[region].sum())
    store[name]['ours_errors'] += int(ours_err[region].sum())


def finalize_pair_stats(store):
    out = {}
    for name, stats in store.items():
        count = stats['count']
        tuned = stats['tuned_errors'] / count if count else float('nan')
        ours = stats['ours_errors'] / count if count else float('nan')
        out[name] = dict(
            voxel_count=count,
            tuned_error_rate=tuned,
            ucrf_udca_error_rate=ours,
            error_reduction=tuned - ours if count else float('nan'))
    return out


def add_region_sums(stats, name, region_bev, r_cam_xy, r_lidar_xy, attn_cam_xy, attn_lidar_xy):
    count = int(region_bev.sum())
    if count == 0:
        return
    stats[name]['bev_count'] += count
    for key, arr in [
            ('r_cam_sum', r_cam_xy),
            ('r_lidar_sum', r_lidar_xy),
            ('attn_cam_sum', attn_cam_xy),
            ('attn_lidar_sum', attn_lidar_xy)]:
        stats[name][key] += float(arr[region_bev].sum())


def attach_region_means(rows, region_mean_stats):
    for row in rows:
        stats = region_mean_stats.get(row['region'], {})
        count = stats.get('bev_count', 0)
        for out_key, sum_key in [
                ('mean_R_cam', 'r_cam_sum'),
                ('mean_R_lidar', 'r_lidar_sum'),
                ('mean_attn_cam', 'attn_cam_sum'),
                ('mean_attn_lidar', 'attn_lidar_sum')]:
            row[out_key] = stats.get(sum_key, 0.0) / count if count else float('nan')
    return rows


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
        raise RuntimeError('CUDA is required for UCRF+UDCA analysis.')

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
    use_official_metric = bool(args.official_style_metric)
    logger.info('Using num_classes=%s ignore_index=%s metric_style=%s',
                num_classes, ignore_index, 'official' if use_official_metric else 'custom')

    ours_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'ucrf_udca')
    tuned_model = build_eval_model(baseline_cfg, args.baseline_ckpt, args.gpu_id, dataset, logger, 'tuned')

    hist_tuned = np.zeros((num_classes, num_classes), dtype=np.float64)
    hist_ours = np.zeros_like(hist_tuned)
    if use_official_metric:
        metric_tuned = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
        metric_ours = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    else:
        metric_tuned = metric_ours = None

    high_stats = defaultdict(lambda: {'count': 0, 'tuned_errors': 0, 'ours_errors': 0})
    visibility_stats = defaultdict(lambda: {'count': 0, 'tuned_errors': 0, 'ours_errors': 0})
    distance_stats_store = defaultdict(lambda: {'count': 0, 'tuned_errors': 0, 'ours_errors': 0})
    region_mean_stats = defaultdict(lambda: defaultdict(float))
    class_gt_counts = np.zeros(num_classes, dtype=np.float64)
    reliability_rows = []
    udca_rows = []
    processed = 0

    for batch_idx, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            tuned_output = tuned_model(return_loss=True, return_result=True, **data_batch)
            ours_output = ours_model(return_loss=True, return_result=True, **data_batch)

        tuned_logits = align_logits_to_bxyzc(extract_logits(tuned_output), num_classes=num_classes)
        ours_logits = align_logits_to_bxyzc(extract_logits(ours_output), num_classes=num_classes)
        spatial_shape = tuple(ours_logits.shape[1:4])
        gt = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'voxel_semantics'),
            target_shape=spatial_shape,
            name='voxel_semantics')
        mask_camera = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_camera'),
            target_shape=spatial_shape,
            name='mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(
            require_batch_tensor(data_batch, 'mask_lidar'),
            target_shape=spatial_shape,
            name='mask_lidar').bool()

        pred_tuned = tuned_logits.softmax(-1).argmax(-1)
        pred_ours = ours_logits.softmax(-1).argmax(-1)
        uncertainty_dict = map_tensor(ours_output, 'uncertainty_dict', {})
        reliability_dict = map_tensor(ours_output, 'reliability_dict', {})
        udca_dict = map_tensor(ours_output, 'udca_dict', {})
        u_total = uncertainty_dict['U_total_voxel']
        u_bev = uncertainty_dict.get('U_total_bev_feat', uncertainty_dict.get('U_total_bev'))
        gate = uncertainty_dict.get('gate_map', None)

        metas = get_img_metas(data_batch)
        batch_size = int(ours_logits.shape[0])
        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            meta = metas[b] if b < len(metas) else {}
            sample_name = sample_name_from_meta(meta, processed)
            file_name = safe_name(sample_name)

            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            tuned_np = tensor_to_numpy(pred_tuned[b]).astype(np.int16)
            ours_np = tensor_to_numpy(pred_ours[b]).astype(np.int16)
            u_np = tensor_to_numpy(u_total[b]).astype(np.float32)
            xy_shape = gt_np.shape[:2]
            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            tuned_err = (tuned_np != gt_np) & valid_mask
            ours_err = (ours_np != gt_np) & valid_mask
            tuned_label_err = (tuned_np != gt_np) & valid_label
            ours_label_err = (ours_np != gt_np) & valid_label
            improved = tuned_err & ~ours_err & valid_mask
            degraded = ~tuned_err & ours_err & valid_mask

            if use_official_metric:
                add_official_metric(metric_tuned, tuned_np, gt_np, mask_lidar_np, mask_camera_np)
                add_official_metric(metric_ours, ours_np, gt_np, mask_lidar_np, mask_camera_np)
            else:
                hist_tuned += hist_info(num_classes, tuned_np, gt_np, valid_mask)
                hist_ours += hist_info(num_classes, ours_np, gt_np, valid_mask)
            metric_count_mask = (mask_camera_np & valid_label) if use_official_metric else valid_mask
            for cls_idx in range(num_classes):
                class_gt_counts[cls_idx] += int(((gt_np == cls_idx) & metric_count_mask).sum())

            r_cam_xy = resize_feature_bev_to_xy(to_numpy_2d(reliability_dict.get('R_cam'), b), xy_shape)
            r_lidar_xy = resize_feature_bev_to_xy(to_numpy_2d(reliability_dict.get('R_lidar'), b), xy_shape)
            discrepancy_xy = resize_feature_bev_to_xy(
                to_numpy_2d(reliability_dict.get('discrepancy_map', reliability_dict.get('D_abs')), b),
                xy_shape)
            visibility_xy = visibility_region_map(mask_camera_np, mask_lidar_np)

            selected_indices = tensor_to_numpy(udca_dict.get('selected_query_indices')[b]) \
                if udca_dict.get('selected_query_indices') is not None else None
            selected_mask_feat = to_numpy_2d(udca_dict.get('selected_query_mask'), b)
            if selected_mask_feat is None:
                selected_xy = np.zeros(xy_shape, dtype=bool)
                feature_hw = (xy_shape[1], xy_shape[0])
            else:
                feature_hw = selected_mask_feat.shape
                selected_xy = feature_mask_to_xy(selected_mask_feat, xy_shape)
            attn_cam_q = tensor_to_numpy(udca_dict.get('attn_weights_cam')[b]).sum(axis=1) \
                if udca_dict.get('attn_weights_cam') is not None else None
            attn_lidar_q = tensor_to_numpy(udca_dict.get('attn_weights_lidar')[b]).sum(axis=1) \
                if udca_dict.get('attn_weights_lidar') is not None else None
            attn_cam_xy = scatter_query_map(selected_indices, attn_cam_q, feature_hw, xy_shape)
            attn_lidar_xy = scatter_query_map(selected_indices, attn_lidar_q, feature_hw, xy_shape)

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(u_np, valid_mask, ratio)
                update_pair_stats(high_stats, key, region, tuned_err, ours_err)

            regions = {
                'visible_by_both': mask_camera_np & mask_lidar_np,
                'camera_only': mask_camera_np & ~mask_lidar_np,
                'lidar_only': ~mask_camera_np & mask_lidar_np,
                'invisible_by_both': ~mask_camera_np & ~mask_lidar_np,
            }
            for name, region in regions.items():
                region_valid = region & valid_label
                update_pair_stats(visibility_stats, name, region_valid, tuned_label_err, ours_label_err)
                region_bev = region.max(axis=2)
                add_region_sums(region_mean_stats, name, region_bev, r_cam_xy, r_lidar_xy, attn_cam_xy, attn_lidar_xy)

            dist_regions = distance_masks(u_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    update_pair_stats(distance_stats_store, name, region & valid_mask, tuned_err, ours_err)

            reliability_rows.append(dict(
                sample_index=processed,
                sample_token=sample_name,
                mean_R_cam=safe_mean(r_cam_xy[np.isfinite(r_cam_xy)]),
                mean_R_lidar=safe_mean(r_lidar_xy[np.isfinite(r_lidar_xy)]),
                mean_discrepancy=safe_mean(discrepancy_xy[np.isfinite(discrepancy_xy)]),
                mean_R_cam_lidar_only=safe_mean(r_cam_xy[(~mask_camera_np & mask_lidar_np).max(axis=2)]),
                mean_R_lidar_lidar_only=safe_mean(r_lidar_xy[(~mask_camera_np & mask_lidar_np).max(axis=2)])))
            offsets_cam = tensor_to_numpy(udca_dict.get('offsets_cam')[b]) \
                if udca_dict.get('offsets_cam') is not None else None
            offsets_lidar = tensor_to_numpy(udca_dict.get('offsets_lidar')[b]) \
                if udca_dict.get('offsets_lidar') is not None else None
            udca_rows.append(dict(
                sample_index=processed,
                sample_token=sample_name,
                num_queries=int(selected_indices.size) if selected_indices is not None else 0,
                num_points=int(offsets_cam.shape[1]) if offsets_cam is not None else 0,
                mean_attn_cam=safe_mean(attn_cam_q) if attn_cam_q is not None else float('nan'),
                mean_attn_lidar=safe_mean(attn_lidar_q) if attn_lidar_q is not None else float('nan'),
                mean_offset_norm_cam=safe_mean(np.linalg.norm(offsets_cam, axis=-1)) if offsets_cam is not None else float('nan'),
                mean_offset_norm_lidar=safe_mean(np.linalg.norm(offsets_lidar, axis=-1)) if offsets_lidar is not None else float('nan'),
                mean_residual_norm=float(tensor_to_numpy(udca_dict.get('udca_residual_norm')))
                if udca_dict.get('udca_residual_norm') is not None else float('nan')))

            if processed < args.vis_topk:
                u_bev_xy = resize_feature_bev_to_xy(to_numpy_2d(u_bev, b), xy_shape)
                gate_xy = resize_feature_bev_to_xy(to_numpy_2d(gate, b), xy_shape)
                plot_grid(
                    [
                        ('R_cam', r_cam_xy, 'viridis', 0.0, 1.0),
                        ('R_lidar', r_lidar_xy, 'viridis', 0.0, 1.0),
                        ('cross-modal discrepancy', discrepancy_xy, 'magma', 0.0, 1.0),
                        ('visibility region', visibility_xy, 'tab10', 0.0, 3.0),
                    ],
                    os.path.join(vis_dir, f'{file_name}_reliability_maps.png'),
                    sample_name)
                plot_grid(
                    [
                        ('U_total_bev', u_bev_xy, 'viridis', 0.0, 1.0),
                        ('selected high-uncertainty queries', selected_xy.astype(float), 'Reds', 0.0, 1.0),
                        ('mean attention to camera', attn_cam_xy, 'viridis', 0.0, max(1e-6, float(np.nanmax(attn_cam_xy)))),
                        ('mean attention to lidar', attn_lidar_xy, 'viridis', 0.0, max(1e-6, float(np.nanmax(attn_lidar_xy)))),
                    ],
                    os.path.join(vis_dir, f'{file_name}_udca_query_attention.png'),
                    sample_name)
                plot_udca_offsets(
                    sample_name,
                    u_bev_xy,
                    selected_xy,
                    offsets_cam,
                    offsets_lidar,
                    os.path.join(vis_dir, f'{file_name}_udca_offsets.png'))
                plot_grid(
                    [
                        ('tuned error map', bev_error(tuned_err), 'Reds', 0.0, 1.0),
                        ('UCRF+UDCA error map', bev_error(ours_err), 'Reds', 0.0, 1.0),
                        ('improved voxels', bev_error(improved), 'Greens', 0.0, 1.0),
                        ('degraded voxels', bev_error(degraded), 'Reds', 0.0, 1.0),
                    ],
                    os.path.join(vis_dir, f'{file_name}_tuned_vs_ucrf_udca_error_overlay.png'),
                    sample_name)
                plot_grid(
                    [
                        ('GT BEV semantic', bev_semantic(gt_np, num_classes - 1), 'tab20', 0, num_classes - 1),
                        ('OUP tuned prediction', bev_semantic(tuned_np, num_classes - 1), 'tab20', 0, num_classes - 1),
                        ('UCRF+UDCA prediction', bev_semantic(ours_np, num_classes - 1), 'tab20', 0, num_classes - 1),
                        ('improved regions', bev_error(improved), 'Greens', 0.0, 1.0),
                    ],
                    os.path.join(vis_dir, f'{file_name}_semantic_comparison.png'),
                    sample_name)
                plot_grid(
                    [
                        ('U_total_bev', u_bev_xy, 'viridis', 0.0, 1.0),
                        ('R_cam', r_cam_xy, 'viridis', 0.0, 1.0),
                        ('R_lidar', r_lidar_xy, 'viridis', 0.0, 1.0),
                        ('refinement gate', gate_xy, 'viridis', 0.0, 1.0),
                        ('corrected voxels', bev_error(improved), 'Greens', 0.0, 1.0),
                    ],
                    os.path.join(vis_dir, f'{file_name}_reliability_uncertainty_refinement.png'),
                    sample_name)

            processed += 1
            logger.info('Processed %s: tuned_err=%.6f ours_err=%.6f',
                        sample_name, safe_rate(tuned_err[valid_mask]), safe_rate(ours_err[valid_mask]))

    if use_official_metric:
        hist_tuned = metric_tuned.hist
        hist_ours = metric_ours.hist

    tuned_miou = miou_from_hist(hist_tuned)
    ours_miou = miou_from_hist(hist_ours)
    tuned_iou = per_class_iou(hist_tuned) * 100.0
    ours_iou = per_class_iou(hist_ours) * 100.0
    per_class_rows = []
    for cls_idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=cls_idx,
            class_name=class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            num_gt_voxels=int(class_gt_counts[cls_idx]),
            tuned_iou=float(tuned_iou[cls_idx]),
            ucrf_udca_iou=float(ours_iou[cls_idx]),
            ucrf_udca_minus_tuned=float(ours_iou[cls_idx] - tuned_iou[cls_idx])))

    high_summary = finalize_pair_stats(high_stats)
    visibility_summary = finalize_pair_stats(visibility_stats)
    distance_summary = finalize_pair_stats(distance_stats_store)
    visibility_rows = attach_region_means(
        [dict(region=k, **v) for k, v in visibility_summary.items()],
        region_mean_stats)
    distance_rows = [dict(region=k, **v) for k, v in distance_summary.items()]
    high_rows = [dict(region=k, **v) for k, v in high_summary.items()]

    reliability_summary = dict(
        mean_R_cam=safe_mean([r['mean_R_cam'] for r in reliability_rows]),
        mean_R_lidar=safe_mean([r['mean_R_lidar'] for r in reliability_rows]),
        mean_discrepancy=safe_mean([r['mean_discrepancy'] for r in reliability_rows]),
        mean_R_cam_lidar_only=safe_mean([r['mean_R_cam_lidar_only'] for r in reliability_rows]),
        mean_R_lidar_lidar_only=safe_mean([r['mean_R_lidar_lidar_only'] for r in reliability_rows]))
    udca_summary = dict(
        num_queries=safe_mean([r['num_queries'] for r in udca_rows]),
        num_points=safe_mean([r['num_points'] for r in udca_rows]),
        mean_attn_cam=safe_mean([r['mean_attn_cam'] for r in udca_rows]),
        mean_attn_lidar=safe_mean([r['mean_attn_lidar'] for r in udca_rows]),
        mean_offset_norm_cam=safe_mean([r['mean_offset_norm_cam'] for r in udca_rows]),
        mean_offset_norm_lidar=safe_mean([r['mean_offset_norm_lidar'] for r in udca_rows]),
        mean_residual_norm=safe_mean([r['mean_residual_norm'] for r in udca_rows]))

    summary = dict(
        overall=dict(
            tuned_mIoU=tuned_miou,
            ucrf_udca_mIoU=ours_miou,
            ucrf_udca_minus_tuned=ours_miou - tuned_miou),
        evaluation_protocol=dict(
            num_samples=processed,
            full_val_size=full_val_size,
            is_full_val=(processed == full_val_size),
            metric_style='official' if use_official_metric else 'custom',
            eval_mask=args.eval_mask,
            metric_mask=metric_mask_name(args.eval_mask, use_official_metric),
            ignore_index=ignore_index,
            include_free_in_mIoU=False),
        high_uncertainty_regions=high_summary,
        visibility_regions=visibility_summary,
        distance_regions=distance_summary,
        per_class_top_improvements=sorted(
            [
                row for row in per_class_rows
                if row['class_index'] != num_classes - 1
                and not math.isnan(row['ucrf_udca_minus_tuned'])
            ],
            key=lambda x: x['ucrf_udca_minus_tuned'],
            reverse=True)[:10],
        reliability_statistics=reliability_summary,
        udca_statistics=udca_summary)

    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(os.path.join(args.out_dir, 'high_uncertainty_improvement.csv'), high_rows)
    write_csv(os.path.join(args.out_dir, 'visibility_region_improvement.csv'), visibility_rows)
    write_csv(os.path.join(args.out_dir, 'distance_region_improvement.csv'), distance_rows)
    write_csv(os.path.join(args.out_dir, 'reliability_statistics.csv'), reliability_rows)
    write_csv(os.path.join(args.out_dir, 'udca_statistics.csv'), udca_rows)
    with open(os.path.join(args.out_dir, 'README_ANALYSIS.md'), 'w') as f:
        f.write(
            '# UCRF + UDCA Analysis\n\n'
            'This directory compares OUP-Occ tuned against OUP-Occ with '
            'Reliability-aware UCRF and sparse uncertainty-guided UDCA.\n\n'
            '- `summary.json`: overall mIoU, region improvements, reliability and UDCA statistics.\n'
            '- `per_class_iou.csv`: tuned vs UCRF+UDCA class IoU.\n'
            '- `vis/`: reliability maps, UDCA queries/offsets, error overlays and semantic comparisons.\n')

    logger.info('Saved UCRF+UDCA analysis to %s', args.out_dir)
    print('\n==== UCRF + UDCA Effect Summary ====')
    print(f'1. tuned mIoU: {tuned_miou:.4f}')
    print(f'2. UCRF+UDCA mIoU: {ours_miou:.4f}')
    print(f'3. UCRF+UDCA - tuned: {ours_miou - tuned_miou:.4f}')
    print(f"4. mean_R_cam / mean_R_lidar: {reliability_summary['mean_R_cam']:.6f} / {reliability_summary['mean_R_lidar']:.6f}")
    print(f"5. mean_attn_cam / mean_attn_lidar: {udca_summary['mean_attn_cam']:.6f} / {udca_summary['mean_attn_lidar']:.6f}")


if __name__ == '__main__':
    main()
