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
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

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
from mmdet3d.models import build_model
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

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze OUP refinement effect')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--out-dir', default='work_dirs/oup_occ_refinement_analysis')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--full-val', action='store_true')
    parser.add_argument('--vis-topk', type=int, default=30)
    parser.add_argument('--eval-mask', default='camera', choices=['camera', 'lidar', 'both', 'union', 'all'])
    parser.add_argument('--official-style-metric', action='store_true', default=True)
    parser.add_argument('--custom-metric', dest='official_style_metric', action='store_false')
    parser.add_argument('--debug-first-batch', action='store_true')
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
    logger = logging.getLogger('refinement_effect')
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


def _checkpoint_state_dict(checkpoint):
    state_dict = checkpoint.get('state_dict', checkpoint)
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def _log_checkpoint_summary(logger, role, model, checkpoint_path, checkpoint):
    state_dict = _checkpoint_state_dict(checkpoint)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    logger.info(
        '%s checkpoint loaded: path=%s, missing_keys=%d, unexpected_keys=%d',
        role, checkpoint_path, len(missing), len(unexpected))
    if missing:
        logger.info('%s missing key examples: %s', role, missing[:10])
    if unexpected:
        logger.info('%s unexpected key examples: %s', role, unexpected[:10])


def build_eval_model(cfg, checkpoint, device_id, dataset, logger, role):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    logger.info('%s model type: %s', role, cfg.model.type)
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, checkpoint, map_location='cpu')
    _log_checkpoint_summary(logger, role, model, checkpoint, ckpt)
    if 'CLASSES' in ckpt.get('meta', {}):
        model.CLASSES = ckpt['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    return model


def hist_info(num_classes, pred, gt, valid):
    pred = pred[valid].astype(np.int64)
    gt = gt[valid].astype(np.int64)
    keep = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    if keep.sum() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.float64)
    return np.bincount(
        num_classes * gt[keep] + pred[keep],
        minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_iou(hist):
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(np.diag(hist), denom, out=np.full_like(np.diag(hist), np.nan), where=denom > 0)


def miou_from_hist(hist):
    iou = per_class_iou(hist)
    if iou.shape[0] > 1:
        return float(np.nanmean(iou[:-1]) * 100.0)
    return float(np.nanmean(iou) * 100.0)


def safe_rate(mask):
    mask = np.asarray(mask)
    if mask.size == 0:
        return float('nan')
    return float(mask.mean())


def safe_mean(values):
    values = np.asarray(values)
    if values.size == 0:
        return float('nan')
    return float(values.mean())


def safe_name(name):
    return str(name).replace('/', '_')


def unique_summary(array, max_items=32):
    values = np.unique(np.asarray(array))
    if values.size > max_items:
        head = ','.join(str(int(v)) for v in values[:max_items])
        return f'[{head},...] ({values.size} unique)'
    return '[' + ','.join(str(int(v)) for v in values) + ']'


def metric_mask_name(eval_mask, official_style):
    if official_style:
        return 'camera'
    return eval_mask


def add_official_metric(metric, pred, gt, mask_lidar, mask_camera):
    # Metric_mIoU mutates arrays in some metric variants, so pass copies.
    metric.add_batch(
        pred.copy(),
        gt.copy(),
        mask_lidar.copy(),
        mask_camera.copy())


def bev_error(error_mask):
    return error_mask.max(axis=2).astype(float)


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


def top_uncertainty_mask(u_total, valid_mask, ratio):
    mask = np.zeros_like(valid_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    if valid_indices.size == 0:
        return mask
    values = u_total.reshape(-1)[valid_indices]
    count = max(1, int(math.ceil(values.size * ratio)))
    top_local = np.argpartition(values, -count)[-count:]
    mask.reshape(-1)[valid_indices[top_local]] = True
    return mask


def bottom_uncertainty_mask(u_total, valid_mask, ratio):
    mask = np.zeros_like(valid_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    if valid_indices.size == 0:
        return mask
    values = u_total.reshape(-1)[valid_indices]
    count = max(1, int(math.ceil(values.size * ratio)))
    bottom_local = np.argpartition(values, count - 1)[:count]
    mask.reshape(-1)[valid_indices[bottom_local]] = True
    return mask


def update_error_stats(store, name, region, baseline_err, coarse_err, final_err, u_total=None):
    count = int(region.sum())
    store[name]['count'] += count
    store[name]['baseline_errors'] += int(baseline_err[region].sum())
    store[name]['coarse_errors'] += int(coarse_err[region].sum())
    store[name]['final_errors'] += int(final_err[region].sum())
    if u_total is not None:
        store[name]['uncertainty_sum'] += float(u_total[region].sum())


def finalize_error_stats(store):
    out = {}
    for name, stats in store.items():
        count = stats['count']
        baseline = stats['baseline_errors'] / count if count else float('nan')
        coarse = stats['coarse_errors'] / count if count else float('nan')
        final = stats['final_errors'] / count if count else float('nan')
        mean_uncertainty = stats.get('uncertainty_sum', 0.0) / count if count else float('nan')
        final_minus_coarse = coarse - final if count else float('nan')
        final_minus_baseline = baseline - final if count else float('nan')
        out[name] = dict(
            voxel_count=count,
            baseline_error_rate=baseline,
            coarse_error_rate=coarse,
            final_error_rate=final,
            error_reduction_from_coarse=final_minus_coarse,
            error_reduction_from_baseline=final_minus_baseline,
            final_minus_coarse_error_reduction=final_minus_coarse,
            final_minus_baseline_error_reduction=final_minus_baseline,
            mean_uncertainty=mean_uncertainty)
    return out


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=True)


def plot_uncertainty_gate(sample_name, u_bev, gate_map, coarse_err, final_err, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    items = [
        ('U_total_bev', u_bev, 'viridis', 0.0, 1.0),
        ('gate_map', gate_map, 'viridis', 0.0, 1.0),
        ('coarse error BEV', bev_error(coarse_err), 'Reds', 0.0, 1.0),
        ('final error BEV', bev_error(final_err), 'Reds', 0.0, 1.0),
    ]
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.flat, items):
        im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_error_overlay(sample_name, coarse_err, final_err, corrected, degraded, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    items = [
        ('coarse prediction error', bev_error(coarse_err)),
        ('final prediction error', bev_error(final_err)),
        ('improved voxels', bev_error(corrected)),
        ('degraded voxels', bev_error(degraded)),
    ]
    for ax, (title, arr) in zip(axes.flat, items):
        im = ax.imshow(arr.T, origin='lower', cmap='Reds', vmin=0.0, vmax=1.0)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_base_vs_tuned_error_overlay(sample_name, base_err, tuned_err, improved, degraded, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    items = [
        ('base prediction error', bev_error(base_err)),
        ('tuned prediction error', bev_error(tuned_err)),
        ('improved voxels', bev_error(improved)),
        ('degraded voxels', bev_error(degraded)),
    ]
    for ax, (title, arr) in zip(axes.flat, items):
        im = ax.imshow(arr.T, origin='lower', cmap='Reds', vmin=0.0, vmax=1.0)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_semantic_comparison(sample_name, gt, baseline, coarse, final, num_classes, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cmap = plt.get_cmap('tab20', num_classes)
    items = [
        ('GT BEV', bev_semantic(gt, num_classes - 1)),
        ('baseline prediction BEV', bev_semantic(baseline, num_classes - 1)),
        ('ours coarse prediction BEV', bev_semantic(coarse, num_classes - 1)),
        ('ours final prediction BEV', bev_semantic(final, num_classes - 1)),
    ]
    for ax, (title, arr) in zip(axes.flat, items):
        im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=0, vmax=num_classes - 1)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_uncertainty(sample_name, gt, coarse, final, top_mask, corrected, num_classes, out_path):
    free_class = num_classes - 1
    cmap = plt.get_cmap('tab20', num_classes)

    def masked_label(label):
        masked = np.full_like(label, free_class)
        masked[top_mask] = label[top_mask]
        return bev_semantic(masked, free_class)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    items = [
        ('top10 GT', masked_label(gt), cmap, 0, num_classes - 1),
        ('top10 coarse prediction', masked_label(coarse), cmap, 0, num_classes - 1),
        ('top10 final prediction', masked_label(final), cmap, 0, num_classes - 1),
        ('corrected region', bev_error(corrected & top_mask), 'Greens', 0.0, 1.0),
    ]
    for ax, (title, arr, cm, vmin, vmax) in zip(axes.flat, items):
        im = ax.imshow(arr.T, origin='lower', cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(f'{sample_name} {title}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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


def print_summary(summary):
    overall = summary['overall']
    protocol = summary.get('evaluation_protocol', {})
    print('\n==== OUP Refinement Effect Summary ====')
    print('0. Evaluation protocol:')
    print(f"   - num samples used: {protocol.get('num_samples')} / {protocol.get('full_val_size')}")
    print(f"   - full validation: {protocol.get('is_full_val')}")
    print(f"   - metric style: {protocol.get('metric_style')}")
    if not protocol.get('is_full_val', False):
        print('   - note: this is subset analysis, not official full-val mIoU.')
    print('1. Overall:')
    print(f"   - baseline mIoU: {overall['baseline_mIoU']:.4f}")
    print(f"   - ours coarse mIoU: {overall['coarse_mIoU']:.4f}")
    print(f"   - ours final mIoU: {overall['final_mIoU']:.4f}")
    print(f"   - final - baseline: {overall['final_minus_baseline']:.4f}")
    print(f"   - final - coarse: {overall['final_minus_coarse']:.4f}")
    print('2. High uncertainty regions:')
    for key in ['top5', 'top10', 'top20']:
        item = summary['high_uncertainty_regions'][key]
        print(f"   - {key}: coarse error {item['coarse_error_rate']:.6f} -> final error {item['final_error_rate']:.6f}, reduction {item['error_reduction_from_coarse']:.6f}")
    print('3. Occlusion regions:')
    for key in ['visible_by_both', 'camera_only', 'lidar_only', 'invisible_by_both']:
        if key in summary['occlusion_regions']:
            item = summary['occlusion_regions'][key]
            print(f"   - {key}: count {item['voxel_count']}, coarse error {item['coarse_error_rate']:.6f} -> final error {item['final_error_rate']:.6f}, mean uncertainty {item['mean_uncertainty']:.6f}")
    print('4. Top per-class improvements:')
    for item in summary['per_class_top_improvements'][:10]:
        print(f"   - {item['class_name']}: final-baseline {item['final_minus_baseline']:.4f}, final-coarse {item['final_minus_coarse']:.4f}")


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
        raise RuntimeError('CUDA is required for refinement effect analysis.')

    dataset_cfg = patch_analysis_pipeline(cfg.data[args.split])
    dataset = build_dataset(dataset_cfg)
    full_val_size = len(dataset)
    sample_limit = None if args.full_val or args.max_samples is None or args.max_samples < 0 else args.max_samples
    is_full_val = sample_limit is None or sample_limit >= full_val_size
    if is_full_val:
        sample_limit = None
    logger.info(
        'Evaluation samples: using %s / %s (%s)',
        full_val_size if sample_limit is None else sample_limit,
        full_val_size,
        'full validation' if is_full_val else 'subset analysis')
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
    if use_official_metric and args.eval_mask != 'camera':
        logger.warning(
            '--official-style-metric follows EFFOcc official camera-mask mIoU; '
            'eval_mask=%s will still be used for explanatory region/top-k analysis.',
            args.eval_mask)
    logger.info(
        'Using num_classes=%s, ignore_index=%s, eval_mask=%s, metric_style=%s',
        num_classes,
        ignore_index,
        args.eval_mask,
        'official' if use_official_metric else 'custom')

    ours_model = build_eval_model(cfg, args.checkpoint, args.gpu_id, dataset, logger, 'ours')
    baseline_model = build_eval_model(baseline_cfg, args.baseline_ckpt, args.gpu_id, dataset, logger, 'baseline')

    hist_baseline = np.zeros((num_classes, num_classes), dtype=np.float64)
    hist_coarse = np.zeros_like(hist_baseline)
    hist_final = np.zeros_like(hist_baseline)
    if use_official_metric:
        metric_baseline = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
        metric_coarse = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
        metric_final = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=True)
    else:
        metric_baseline = metric_coarse = metric_final = None

    stats_factory = lambda: {
        'count': 0,
        'baseline_errors': 0,
        'coarse_errors': 0,
        'final_errors': 0,
        'uncertainty_sum': 0.0
    }
    high_unc_stats = defaultdict(stats_factory)
    low_unc_stats = defaultdict(stats_factory)
    occ_region_stats = defaultdict(stats_factory)
    distance_stats = defaultdict(stats_factory)
    per_sample_rows = []
    class_gt_counts = np.zeros(num_classes, dtype=np.float64)

    processed = 0
    for batch_idx, data_batch in enumerate(data_loader):
        if sample_limit is not None and processed >= sample_limit:
            break
        with torch.no_grad():
            baseline_logits = baseline_model(return_loss=True, return_result=True, **data_batch)
            ours_output = ours_model(return_loss=True, return_result=True, **data_batch)

        baseline_logits = align_logits_to_bxyzc(baseline_logits, num_classes=num_classes)
        coarse_logits = align_logits_to_bxyzc(ours_output['coarse_occ_logits'], num_classes=num_classes)
        final_logits = align_logits_to_bxyzc(ours_output['final_occ_logits'], num_classes=num_classes)
        spatial_shape = tuple(final_logits.shape[1:4])

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

        pred_baseline = baseline_logits.softmax(-1).argmax(-1)
        pred_coarse = coarse_logits.softmax(-1).argmax(-1)
        pred_final = final_logits.softmax(-1).argmax(-1)
        u_total = ours_output['uncertainty_dict']['U_total_voxel']
        gate = ours_output['uncertainty_dict']['gate_map']
        u_bev = ours_output['uncertainty_dict']['U_total_bev']

        if batch_idx == 0:
            logger.info('baseline logits shape: %s', tuple(baseline_logits.shape))
            logger.info('coarse logits shape: %s', tuple(coarse_logits.shape))
            logger.info('final logits shape: %s', tuple(final_logits.shape))
            logger.info('U_total_voxel shape: %s', tuple(u_total.shape))
            logger.info('gate_map shape: %s', tuple(gate.shape))
            logger.info(
                'prediction class ranges: baseline [%s, %s], coarse [%s, %s], final [%s, %s]',
                int(pred_baseline.min()), int(pred_baseline.max()),
                int(pred_coarse.min()), int(pred_coarse.max()),
                int(pred_final.min()), int(pred_final.max()))

        metas = get_img_metas(data_batch)
        batch_size = int(final_logits.shape[0])
        for b in range(batch_size):
            if sample_limit is not None and processed >= sample_limit:
                break
            meta = metas[b] if b < len(metas) else {}
            sample_name = sample_name_from_meta(meta, processed)
            file_name = safe_name(sample_name)

            gt_np = tensor_to_numpy(gt[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            baseline_np = tensor_to_numpy(pred_baseline[b]).astype(np.int16)
            coarse_np = tensor_to_numpy(pred_coarse[b]).astype(np.int16)
            final_np = tensor_to_numpy(pred_final[b]).astype(np.int16)
            u_np = tensor_to_numpy(u_total[b]).astype(np.float32)
            u_bev_np = tensor_to_numpy(u_bev[b]).astype(np.float32)
            gate_np = tensor_to_numpy(gate[b]).astype(np.float32)
            if gate_np.ndim == 3:
                gate_np = gate_np.mean(axis=0)

            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            baseline_err = (baseline_np != gt_np) & valid_mask
            coarse_err = (coarse_np != gt_np) & valid_mask
            final_err = (final_np != gt_np) & valid_mask
            baseline_label_err = (baseline_np != gt_np) & valid_label
            coarse_label_err = (coarse_np != gt_np) & valid_label
            final_label_err = (final_np != gt_np) & valid_label
            corrected = coarse_err & ~final_err & valid_mask
            degraded = ~coarse_err & final_err & valid_mask
            improved_vs_baseline = baseline_err & ~final_err & valid_mask
            degraded_vs_baseline = ~baseline_err & final_err & valid_mask

            if use_official_metric:
                add_official_metric(metric_baseline, baseline_np, gt_np, mask_lidar_np, mask_camera_np)
                add_official_metric(metric_coarse, coarse_np, gt_np, mask_lidar_np, mask_camera_np)
                add_official_metric(metric_final, final_np, gt_np, mask_lidar_np, mask_camera_np)
            else:
                hist_baseline += hist_info(num_classes, baseline_np, gt_np, valid_mask)
                hist_coarse += hist_info(num_classes, coarse_np, gt_np, valid_mask)
                hist_final += hist_info(num_classes, final_np, gt_np, valid_mask)
            metric_count_mask = (mask_camera_np & valid_label) if use_official_metric else valid_mask
            for cls_idx in range(num_classes):
                class_gt_counts[cls_idx] += int(((gt_np == cls_idx) & metric_count_mask).sum())

            visible_by_both = mask_camera_np & mask_lidar_np
            camera_only = mask_camera_np & ~mask_lidar_np
            lidar_only = ~mask_camera_np & mask_lidar_np
            invisible_by_both = ~mask_camera_np & ~mask_lidar_np
            regions = {
                'visible_by_both': visible_by_both,
                'camera_only': camera_only,
                'lidar_only': lidar_only,
                'invisible_by_both': invisible_by_both,
            }

            if args.debug_first_batch and batch_idx == 0 and b == 0:
                logger.info('==== Debug first batch/sample ====')
                logger.info('sample index: %s, token: %s', processed, sample_name)
                logger.info('gt shape: %s', gt_np.shape)
                logger.info('mask_camera shape: %s', mask_camera_np.shape)
                logger.info('mask_lidar shape: %s', mask_lidar_np.shape)
                logger.info('baseline logits shape: %s', tuple(baseline_logits.shape))
                logger.info('coarse logits shape: %s', tuple(coarse_logits.shape))
                logger.info('final logits shape: %s', tuple(final_logits.shape))
                logger.info('argmax dim: -1')
                logger.info('baseline pred unique classes: %s', unique_summary(baseline_np))
                logger.info('coarse pred unique classes: %s', unique_summary(coarse_np))
                logger.info('final pred unique classes: %s', unique_summary(final_np))
                logger.info('gt unique classes: %s', unique_summary(gt_np[valid_label]))
                logger.info('valid voxel count (%s): %d', args.eval_mask, int(valid_mask.sum()))
                for region_name, region_mask in regions.items():
                    logger.info(
                        '%s count on valid_label: %d',
                        region_name,
                        int((region_mask & valid_label).sum()))

            row = dict(
                sample_index=processed,
                sample_token=sample_name,
                num_valid_voxels=int(valid_mask.sum()),
                baseline_error_rate=safe_rate(baseline_err[valid_mask]),
                coarse_error_rate=safe_rate(coarse_err[valid_mask]),
                final_error_rate=safe_rate(final_err[valid_mask]),
                corrected_voxels=int(corrected.sum()),
                degraded_voxels=int(degraded.sum()))

            for ratio, key in [(0.05, 'top5'), (0.10, 'top10'), (0.20, 'top20')]:
                region = top_uncertainty_mask(u_np, valid_mask, ratio)
                update_error_stats(high_unc_stats, key, region, baseline_err, coarse_err, final_err, u_np)
                row[f'{key}_voxel_count'] = int(region.sum())
                row[f'{key}_baseline_error_rate'] = safe_rate(baseline_err[region])
                row[f'{key}_coarse_error_rate'] = safe_rate(coarse_err[region])
                row[f'{key}_final_error_rate'] = safe_rate(final_err[region])
                row[f'{key}_error_reduction_from_coarse'] = (
                    row[f'{key}_coarse_error_rate'] - row[f'{key}_final_error_rate'])

            low_region = bottom_uncertainty_mask(u_np, valid_mask, 0.50)
            update_error_stats(low_unc_stats, 'bottom50', low_region, baseline_err, coarse_err, final_err, u_np)
            row['bottom50_baseline_error_rate'] = safe_rate(baseline_err[low_region])
            row['bottom50_final_error_rate'] = safe_rate(final_err[low_region])

            for name, region in regions.items():
                region_valid = region & valid_label
                update_error_stats(
                    occ_region_stats,
                    name,
                    region_valid,
                    baseline_label_err,
                    coarse_label_err,
                    final_label_err,
                    u_np)
                row[f'{name}_voxel_count'] = int(region_valid.sum())
                row[f'{name}_mean_uncertainty'] = safe_mean(u_np[region_valid])
                row[f'{name}_coarse_error_rate'] = safe_rate(coarse_label_err[region_valid])
                row[f'{name}_final_error_rate'] = safe_rate(final_label_err[region_valid])

            dist_regions = distance_masks(u_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    update_error_stats(
                        distance_stats, name, region & valid_mask, baseline_err, coarse_err, final_err, u_np)

            if processed < args.vis_topk:
                plot_uncertainty_gate(
                    sample_name,
                    u_bev_np,
                    gate_np,
                    coarse_err,
                    final_err,
                    os.path.join(vis_dir, f'{file_name}_uncertainty_gate.png'))
                plot_error_overlay(
                    sample_name,
                    coarse_err,
                    final_err,
                    corrected,
                    degraded,
                    os.path.join(vis_dir, f'{file_name}_coarse_final_error_overlay.png'))
                plot_base_vs_tuned_error_overlay(
                    sample_name,
                    baseline_err,
                    final_err,
                    improved_vs_baseline,
                    degraded_vs_baseline,
                    os.path.join(vis_dir, f'{file_name}_base_vs_tuned_error_overlay.png'))
                plot_semantic_comparison(
                    sample_name,
                    gt_np,
                    baseline_np,
                    coarse_np,
                    final_np,
                    num_classes,
                    os.path.join(vis_dir, f'{file_name}_semantic_comparison.png'))
                plot_top_uncertainty(
                    sample_name,
                    gt_np,
                    coarse_np,
                    final_np,
                    top_uncertainty_mask(u_np, valid_mask, 0.10),
                    corrected,
                    num_classes,
                    os.path.join(vis_dir, f'{file_name}_top_uncertainty_refinement.png'))

            per_sample_rows.append(row)
            processed += 1
            logger.info(
                'Processed %s: baseline %.6f, coarse %.6f, final %.6f',
                sample_name,
                row['baseline_error_rate'],
                row['coarse_error_rate'],
                row['final_error_rate'])

    if use_official_metric:
        hist_baseline = metric_baseline.hist
        hist_coarse = metric_coarse.hist
        hist_final = metric_final.hist

    baseline_miou = miou_from_hist(hist_baseline)
    coarse_miou = miou_from_hist(hist_coarse)
    final_miou = miou_from_hist(hist_final)
    baseline_iou = per_class_iou(hist_baseline) * 100.0
    coarse_iou = per_class_iou(hist_coarse) * 100.0
    final_iou = per_class_iou(hist_final) * 100.0

    per_class_rows = []
    for cls_idx in range(num_classes):
        per_class_rows.append(dict(
            class_index=cls_idx,
            class_name=class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            num_gt_voxels=int(class_gt_counts[cls_idx]),
            baseline_iou=float(baseline_iou[cls_idx]),
            coarse_iou=float(coarse_iou[cls_idx]),
            final_iou=float(final_iou[cls_idx]),
            final_minus_baseline=float(final_iou[cls_idx] - baseline_iou[cls_idx]),
            final_minus_coarse=float(final_iou[cls_idx] - coarse_iou[cls_idx])))

    region_summary = finalize_error_stats(occ_region_stats)
    distance_summary = finalize_error_stats(distance_stats)
    high_summary = finalize_error_stats(high_unc_stats)
    low_summary = finalize_error_stats(low_unc_stats)
    for name, item in region_summary.items():
        if item['voxel_count'] == 0:
            logger.warning(
                'Occlusion region %s has voxel_count=0 on valid labels. '
                'Check mask_camera/mask_lidar fields and dataset split.',
                name)

    protocol = dict(
        num_samples=processed,
        full_val_size=full_val_size,
        is_full_val=(processed == full_val_size),
        metric_style='official' if use_official_metric else 'custom',
        eval_mask=args.eval_mask,
        metric_mask=metric_mask_name(args.eval_mask, use_official_metric),
        ignore_index=ignore_index,
        include_free_in_mIoU=False,
        class_names=class_names,
        note='This is subset analysis, not official full-val mIoU.'
        if processed != full_val_size else 'This analysis used the full validation set.')

    summary = dict(
        overall=dict(
            baseline_mIoU=baseline_miou,
            coarse_mIoU=coarse_miou,
            final_mIoU=final_miou,
            baseline_subset_mIoU=baseline_miou,
            coarse_subset_mIoU=coarse_miou,
            final_subset_mIoU=final_miou,
            baseline_full_val_mIoU=baseline_miou if processed == full_val_size else None,
            coarse_full_val_mIoU=coarse_miou if processed == full_val_size else None,
            final_full_val_mIoU=final_miou if processed == full_val_size else None,
            final_minus_baseline=final_miou - baseline_miou,
            final_minus_coarse=final_miou - coarse_miou),
        evaluation_protocol=protocol,
        high_uncertainty_regions=high_summary,
        low_uncertainty_regions=low_summary,
        occlusion_regions=region_summary,
        distance=distance_summary,
        per_class=per_class_rows,
        per_class_top_improvements=sorted(
            [
                row for row in per_class_rows
                if row['class_index'] != num_classes - 1
                and not math.isnan(row['final_minus_baseline'])
            ],
            key=lambda x: x['final_minus_baseline'],
            reverse=True)[:10],
        num_samples=processed,
        eval_mask=args.eval_mask)

    save_json(os.path.join(args.out_dir, 'summary.json'), summary)
    write_csv(os.path.join(args.out_dir, 'per_sample_metrics.csv'), per_sample_rows)
    write_csv(os.path.join(args.out_dir, 'per_class_iou.csv'), per_class_rows)
    write_csv(
        os.path.join(args.out_dir, 'region_improvement.csv'),
        [dict(region=k, **v) for k, v in region_summary.items()])
    write_csv(
        os.path.join(args.out_dir, 'distance_improvement.csv'),
        [dict(distance=k, **v) for k, v in distance_summary.items()])
    logger.info('Saved refinement analysis to %s', args.out_dir)
    print_summary(summary)


if __name__ == '__main__':
    main()
