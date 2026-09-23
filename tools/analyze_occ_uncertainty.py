#!/usr/bin/env python
import argparse
import copy
import csv
import json
import logging
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
from mmcv.parallel import DataContainer as DC
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
from mmdet3d.models.occ_uncertainty import (
    align_logits_to_bxyzc,
    align_tensor_to_bxyz,
    compute_confidence_uncertainty,
    compute_cross_modal_uncertainty,
    compute_semantic_entropy,
    compute_total_uncertainty,
    compute_visibility_uncertainty,
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Post-hoc occupancy uncertainty analysis')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', default='work_dirs/occ_uncertainty_analysis')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--save-npz', action='store_true')
    parser.add_argument('--vis-topk', type=int, default=20)
    parser.add_argument('--bev-proj', default='max', choices=['max', 'mean'])
    parser.add_argument('--eval-mask', default='camera', choices=['camera', 'lidar', 'both', 'union', 'all'])
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--ignore-index', type=int, default=None)
    parser.add_argument('--ignore-free-class', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def setup_logger(out_dir):
    mmcv.mkdir_or_exist(out_dir)
    logger = logging.getLogger('occ_uncertainty')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(out_dir, 'analysis.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def patch_analysis_pipeline(dataset_cfg):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    pipeline = []
    for step in dataset_cfg.pipeline:
        step = copy.deepcopy(step)
        if step.get('type') == 'MultiScaleFlipAug3D':
            pipeline.extend(copy.deepcopy(step.get('transforms', [])))
        else:
            pipeline.append(step)

    has_occ_loader = any(step.get('type') == 'LoadOccGTFromFile' for step in pipeline)
    if not has_occ_loader:
        insert_at = 0
        for idx, step in enumerate(pipeline):
            if step.get('type') == 'ToEgo':
                insert_at = idx + 1
                break
            if step.get('type') in ['BEVAug', 'PointToMultiViewDepthFusion']:
                insert_at = idx
                break
        pipeline.insert(insert_at, dict(type='LoadOccGTFromFile'))

    collect_keys = ['points', 'img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera']
    has_collect = False
    for step in pipeline:
        if step.get('type') == 'PrepareImageInputs':
            step['is_train'] = False
        if step.get('type') == 'BEVAug':
            step['is_train'] = False
        if step.get('type') == 'DefaultFormatBundle3D':
            step['with_label'] = False
        if step.get('type') == 'Collect3D':
            has_collect = True
            keys = list(step.get('keys', []))
            for key in collect_keys:
                if key not in keys:
                    keys.append(key)
            step['keys'] = keys

    if not has_collect:
        pipeline.append(dict(type='Collect3D', keys=collect_keys))

    dataset_cfg.pipeline = pipeline
    dataset_cfg.test_mode = True
    return dataset_cfg


def unwrap_data(value):
    if isinstance(value, DC):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_data(value[0])
    return value


def require_batch_tensor(data_batch, key):
    if key not in data_batch:
        raise KeyError(f'{key} is not found in data_batch. Please check Collect3D fields in config.')
    value = unwrap_data(data_batch[key])
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{key} must be tensor-like after collate, got {type(value)}')
    return value


def get_img_metas(data_batch):
    metas = unwrap_data(data_batch.get('img_metas'))
    while isinstance(metas, (list, tuple)) and len(metas) == 1 and not isinstance(metas[0], dict):
        metas = metas[0]
    if isinstance(metas, dict):
        return [metas]
    if isinstance(metas, (list, tuple)) and (len(metas) == 0 or isinstance(metas[0], dict)):
        return list(metas)
    return []


def sample_name_from_meta(meta, index):
    for key in ['sample_token', 'token', 'sample_idx']:
        if key in meta and meta[key] is not None:
            return str(meta[key])
    return f'sample_{index:06d}'


def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def safe_mean(values):
    values = np.asarray(values)
    if values.size == 0:
        return float('nan')
    return float(values.mean())


def safe_rate(mask):
    mask = np.asarray(mask)
    if mask.size == 0:
        return float('nan')
    return float(mask.mean())


def bev_project(volume, mode='max'):
    if mode == 'mean':
        return np.nanmean(volume, axis=2)
    return np.nanmax(volume, axis=2)


def save_heatmap(array, path, title, cmap='viridis', vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(array.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_component_heatmaps(sample_name, u_sem, u_conf, u_occ, u_total, out_path, mode='max'):
    components = [
        ('U_sem', bev_project(u_sem, mode)),
        ('U_conf', bev_project(u_conf, mode)),
        ('U_occ', bev_project(u_occ, mode)),
        ('U_total', bev_project(u_total, mode)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (title, array) in zip(axes.flat, components):
        im = ax.imshow(array.T, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(f'{sample_name} {title}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_error_overlay(sample_name, u_total, error_mask, out_path, mode='max'):
    u_bev = bev_project(u_total, mode)
    error_bev = error_mask.max(axis=2).astype(bool)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(u_bev.T, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
    overlay = np.ma.masked_where(~error_bev.T, error_bev.T)
    ax.imshow(overlay, origin='lower', cmap='Reds', alpha=0.45)
    ax.set_title(f'{sample_name} uncertainty with error overlay')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_occlusion_masks(sample_name, mask_camera, mask_lidar, u_occ, out_path, mode='max'):
    components = [
        ('mask_camera', bev_project(mask_camera.astype(float), 'max')),
        ('mask_lidar', bev_project(mask_lidar.astype(float), 'max')),
        ('U_occ', bev_project(u_occ, mode)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, array) in zip(axes.flat, components):
        im = ax.imshow(array.T, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(f'{sample_name} {title}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def confusion_iou(pred, gt, valid, num_classes, cls_idx):
    pred_cls = (pred == cls_idx) & valid
    gt_cls = (gt == cls_idx) & valid
    union = pred_cls | gt_cls
    if union.sum() == 0:
        return float('nan')
    return float((pred_cls & gt_cls).sum() / union.sum())


def get_ignore_index(cfg, args):
    if args.ignore_index is not None:
        return args.ignore_index
    for head_name in ['occ_head', 'final_occ_head', 'coarse_occ_head']:
        try:
            return int(getattr(cfg.model, head_name).loss_occ.get('ignore_index', 255))
        except Exception:
            pass
    return 255


def get_num_classes(cfg, occ_logits=None):
    for head_name in ['occ_head', 'final_occ_head', 'coarse_occ_head']:
        try:
            num_classes = getattr(cfg.model, head_name).num_classes
            if num_classes is not None:
                return int(num_classes)
        except Exception:
            pass
    if occ_logits is not None:
        return int(occ_logits.shape[-1])
    return None


def get_class_names(num_classes):
    try:
        names = Metric_mIoU(num_classes=num_classes).class_names
        if len(names) >= num_classes:
            return names[:num_classes]
    except Exception:
        pass
    return [str(i) for i in range(num_classes)]


def build_valid_mask(gt, mask_camera, mask_lidar, num_classes, ignore_index, eval_mask):
    label_valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    if eval_mask == 'camera':
        eval_valid = mask_camera
    elif eval_mask == 'lidar':
        eval_valid = mask_lidar
    elif eval_mask == 'both':
        eval_valid = mask_camera & mask_lidar
    elif eval_mask == 'union':
        eval_valid = mask_camera | mask_lidar
    else:
        eval_valid = np.ones_like(label_valid, dtype=bool)
    return label_valid & eval_valid, label_valid


def topk_error_rates(u_total, error_mask, valid_mask, ks=(0.05, 0.10, 0.20)):
    valid_u = u_total[valid_mask]
    valid_error = error_mask[valid_mask]
    out = {}
    if valid_u.size == 0:
        for k in ks:
            name = f'top{int(k * 100)}'
            out[name] = {'error_rate': float('nan'), 'error_count': 0, 'voxel_count': 0}
        return out
    order = np.argsort(-valid_u)
    for k in ks:
        count = max(1, int(math.ceil(valid_u.size * k)))
        top_idx = order[:count]
        name = f'top{int(k * 100)}'
        out[name] = {
            'error_rate': safe_rate(valid_error[top_idx]),
            'error_count': int(valid_error[top_idx].sum()),
            'voxel_count': int(count),
        }
    return out


def distance_masks(shape, cfg, logger):
    xdim, ydim, zdim = shape
    point_cloud_range = cfg.get('point_cloud_range', None)
    if point_cloud_range is None or len(point_cloud_range) < 5:
        logger.warning('Distance-based uncertainty is skipped: point_cloud_range is not available in config.')
        return None

    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in point_cloud_range]
    x_step = (x_max - x_min) / float(xdim)
    y_step = (y_max - y_min) / float(ydim)
    xs = x_min + (np.arange(xdim) + 0.5) * x_step
    ys = y_min + (np.arange(ydim) + 0.5) * y_step
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    dist = np.sqrt(xx ** 2 + yy ** 2)
    dist = np.repeat(dist[:, :, None], zdim, axis=2)
    return {
        '0-15m': dist < 15.0,
        '15-30m': (dist >= 15.0) & (dist < 30.0),
        '30-45m': (dist >= 30.0) & (dist < 45.0),
        '45m+': dist >= 45.0,
    }


def add_sum_count(store, key, values):
    values = np.asarray(values)
    if values.size == 0:
        return
    store[key]['sum'] += float(values.sum())
    store[key]['count'] += int(values.size)


def mean_from_sum_count(item):
    if item['count'] == 0:
        return float('nan')
    return item['sum'] / item['count']


def update_region_stats(store, name, region_mask, valid_label, error_mask, u_total):
    region_valid = region_mask & valid_label
    store[name]['voxel_count'] += int(region_valid.sum())
    store[name]['uncertainty_sum'] += float(u_total[region_valid].sum())
    store[name]['error_sum'] += int(error_mask[region_valid].sum())


def finalize_region_stats(store):
    out = {}
    for name, stats in store.items():
        count = stats['voxel_count']
        out[name] = {
            'voxel_count': count,
            'mean_U_total': stats['uncertainty_sum'] / count if count else float('nan'),
            'error_rate': stats['error_sum'] / count if count else float('nan'),
        }
    return out


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def print_summary(summary):
    print('\n==== Occupancy Uncertainty Summary ====')
    error_mean = summary['error_correlation']['mean_U_total_on_error_voxels']
    correct_mean = summary['error_correlation']['mean_U_total_on_correct_voxels']
    ratio = summary['error_correlation']['U_total_error_correct_ratio']
    print('1. Mean uncertainty on error voxels vs correct voxels:')
    print(f'   - U_total error mean: {error_mean:.6f}')
    print(f'   - U_total correct mean: {correct_mean:.6f}')
    print(f'   - ratio: {ratio:.6f}')
    print('2. Top-k uncertainty error concentration:')
    topk = summary['topk_error_rates']
    print(f"   - global error rate: {topk['error_rate_all_valid_voxels']:.6f}")
    print(f"   - top 5% uncertainty error rate: {topk['error_rate_top5_uncertainty']:.6f}")
    print(f"   - top 10% uncertainty error rate: {topk['error_rate_top10_uncertainty']:.6f}")
    print(f"   - top 20% uncertainty error rate: {topk['error_rate_top20_uncertainty']:.6f}")
    print('3. Occlusion region statistics:')
    for name, stats in summary['occlusion_regions'].items():
        print(f"   - {name}: mean uncertainty={stats['mean_U_total']:.6f}, error rate={stats['error_rate']:.6f}")
    print('4. Per-class uncertainty ranking:')
    for item in summary['per_class_uncertainty_top10']:
        print(f"   - {item['class_name']}: mean_U_total={item['mean_U_total']:.6f}, error_rate={item['error_rate']:.6f}")


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    set_random_seed(args.seed, deterministic=False)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [args.gpu_id]

    split_cfg = patch_analysis_pipeline(cfg.data[args.split])
    dataset = build_dataset(split_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for EFFOcc model inference in this analysis script.')
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()

    npz_dir = os.path.join(args.out_dir, 'npz')
    vis_dir = os.path.join(args.out_dir, 'vis')
    if args.save_npz:
        mmcv.mkdir_or_exist(npz_dir)
    mmcv.mkdir_or_exist(vis_dir)

    ignore_index = get_ignore_index(cfg, args)
    num_classes = get_num_classes(cfg)
    if num_classes is None:
        raise ValueError('Cannot infer num_classes from cfg.model.occ_head.num_classes.')
    class_names = get_class_names(num_classes)
    free_class = num_classes - 1
    ignore_classes = [free_class] if args.ignore_free_class else None

    logger.info('Using ignore_index=%s, num_classes=%s, eval_mask=%s', ignore_index, num_classes, args.eval_mask)
    logger.info('Patched analysis pipeline: %s', [step.get('type') for step in split_cfg.pipeline])

    component_acc = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    topk_error_count = defaultdict(int)
    topk_voxel_count = defaultdict(int)
    occlusion_region_acc = defaultdict(lambda: {'voxel_count': 0, 'uncertainty_sum': 0.0, 'error_sum': 0})
    class_acc = defaultdict(lambda: {'count': 0, 'uncertainty_sum': 0.0, 'error_sum': 0, 'intersection': 0, 'union': 0})
    distance_acc = defaultdict(lambda: {'voxel_count': 0, 'uncertainty_sum': 0.0, 'error_sum': 0})
    per_sample_rows = []
    sample_count = 0
    total_valid_errors = 0
    total_valid_voxels = 0
    fusion_weights = None

    for batch_idx, data_batch in enumerate(data_loader):
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break

        with torch.no_grad():
            occ_logits_raw = model(return_loss=True, return_result=True, **data_batch)

        gt_raw = require_batch_tensor(data_batch, 'voxel_semantics')
        mask_camera_raw = require_batch_tensor(data_batch, 'mask_camera')
        mask_lidar_raw = require_batch_tensor(data_batch, 'mask_lidar')

        occ_logits = align_logits_to_bxyzc(occ_logits_raw, num_classes=num_classes)
        spatial_shape = tuple(occ_logits.shape[1:4])
        gt_label = align_tensor_to_bxyz(gt_raw, target_shape=spatial_shape, name='voxel_semantics').long()
        mask_camera = align_tensor_to_bxyz(mask_camera_raw, target_shape=spatial_shape, name='mask_camera').bool()
        mask_lidar = align_tensor_to_bxyz(mask_lidar_raw, target_shape=spatial_shape, name='mask_lidar').bool()

        occ_prob = torch.softmax(occ_logits.float(), dim=-1)
        pred_label = occ_prob.argmax(dim=-1)
        u_sem = compute_semantic_entropy(occ_prob=occ_prob, ignore_classes=ignore_classes)
        u_conf = compute_confidence_uncertainty(occ_prob=occ_prob)
        u_occ = compute_visibility_uncertainty(mask_camera, mask_lidar)
        u_cm = compute_cross_modal_uncertainty(
            None, None, target_shape=spatial_shape, logger=logger if batch_idx == 0 else None)
        u_total, fusion_weights = compute_total_uncertainty(
            u_sem,
            u_occ,
            u_conf,
            u_cm=u_cm,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
        )

        if batch_idx == 0:
            logger.info('occ_logits aligned shape: %s', tuple(occ_logits.shape))
            logger.info('occ_prob shape: %s', tuple(occ_prob.shape))
            logger.info('pred_label shape: %s', tuple(pred_label.shape))
            logger.info('gt_label shape: %s', tuple(gt_label.shape))
            logger.info('mask_camera shape: %s', tuple(mask_camera.shape))
            logger.info('mask_lidar shape: %s', tuple(mask_lidar.shape))
            logger.info('fusion weights: %s', fusion_weights)

        metas = get_img_metas(data_batch)
        batch_size = int(occ_logits.shape[0])
        for b in range(batch_size):
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break
            meta = metas[b] if b < len(metas) else {}
            sample_name = sample_name_from_meta(meta, sample_count)
            safe_sample_name = sample_name.replace('/', '_')

            pred_np = tensor_to_numpy(pred_label[b]).astype(np.int16)
            gt_np = tensor_to_numpy(gt_label[b]).astype(np.int16)
            mask_camera_np = tensor_to_numpy(mask_camera[b]).astype(bool)
            mask_lidar_np = tensor_to_numpy(mask_lidar[b]).astype(bool)
            u_sem_np = tensor_to_numpy(u_sem[b]).astype(np.float32)
            u_conf_np = tensor_to_numpy(u_conf[b]).astype(np.float32)
            u_occ_np = tensor_to_numpy(u_occ[b]).astype(np.float32)
            u_total_np = tensor_to_numpy(u_total[b]).astype(np.float32)

            valid_mask, valid_label = build_valid_mask(
                gt_np, mask_camera_np, mask_lidar_np, num_classes, ignore_index, args.eval_mask)
            error_mask = (pred_np != gt_np) & valid_mask
            correct_mask = (pred_np == gt_np) & valid_mask
            global_error_rate = safe_rate(error_mask[valid_mask])

            row = {
                'sample_index': sample_count,
                'sample_token': sample_name,
                'num_valid_voxels': int(valid_mask.sum()),
                'error_rate_all_valid_voxels': global_error_rate,
            }
            for comp_name, comp in [
                ('U_total', u_total_np),
                ('U_sem', u_sem_np),
                ('U_occ', u_occ_np),
                ('U_conf', u_conf_np),
            ]:
                err_key = f'mean_{comp_name}_on_error_voxels'
                cor_key = f'mean_{comp_name}_on_correct_voxels'
                row[err_key] = safe_mean(comp[error_mask])
                row[cor_key] = safe_mean(comp[correct_mask])
                add_sum_count(component_acc, err_key, comp[error_mask])
                add_sum_count(component_acc, cor_key, comp[correct_mask])

            topk = topk_error_rates(u_total_np, error_mask, valid_mask)
            for key, stats in topk.items():
                row[f'error_rate_{key}_uncertainty'] = stats['error_rate']
                topk_error_count[key] += stats['error_count']
                topk_voxel_count[key] += stats['voxel_count']

            total_valid_errors += int(error_mask.sum())
            total_valid_voxels += int(valid_mask.sum())

            visible_by_both = mask_camera_np & mask_lidar_np
            camera_only = mask_camera_np & ~mask_lidar_np
            lidar_only = ~mask_camera_np & mask_lidar_np
            invisible_by_both = ~mask_camera_np & ~mask_lidar_np
            for name, region in [
                ('visible_by_both', visible_by_both),
                ('camera_only', camera_only),
                ('lidar_only', lidar_only),
                ('invisible_by_both', invisible_by_both),
            ]:
                update_region_stats(occlusion_region_acc, name, region, valid_label, pred_np != gt_np, u_total_np)
                region_valid = region & valid_label
                row[f'{name}_voxel_count'] = int(region_valid.sum())
                row[f'{name}_mean_U_total'] = safe_mean(u_total_np[region_valid])
                row[f'{name}_error_rate'] = safe_rate((pred_np != gt_np)[region_valid])

            for cls_idx in range(num_classes):
                cls_valid = valid_mask & (gt_np == cls_idx)
                cls_pred = valid_mask & (pred_np == cls_idx)
                cls_intersection = int((cls_valid & cls_pred).sum())
                cls_union = int((cls_valid | cls_pred).sum())
                class_acc[cls_idx]['count'] += int(cls_valid.sum())
                class_acc[cls_idx]['uncertainty_sum'] += float(u_total_np[cls_valid].sum())
                class_acc[cls_idx]['error_sum'] += int((pred_np != gt_np)[cls_valid].sum())
                class_acc[cls_idx]['intersection'] += cls_intersection
                class_acc[cls_idx]['union'] += cls_union

            dist_regions = distance_masks(u_total_np.shape, cfg, logger)
            if dist_regions is not None:
                for name, region in dist_regions.items():
                    region_valid = region & valid_mask
                    distance_acc[name]['voxel_count'] += int(region_valid.sum())
                    distance_acc[name]['uncertainty_sum'] += float(u_total_np[region_valid].sum())
                    distance_acc[name]['error_sum'] += int(error_mask[region_valid].sum())

            if args.save_npz:
                np.savez_compressed(
                    os.path.join(npz_dir, f'{safe_sample_name}.npz'),
                    pred_label=pred_np.astype(np.uint8),
                    gt_label=gt_np,
                    mask_camera=mask_camera_np,
                    mask_lidar=mask_lidar_np,
                    U_sem=u_sem_np.astype(np.float16),
                    U_conf=u_conf_np.astype(np.float16),
                    U_occ=u_occ_np.astype(np.float16),
                    U_total=u_total_np.astype(np.float16),
                    sample_token=sample_name,
                )

            if sample_count < args.vis_topk:
                save_heatmap(
                    bev_project(u_total_np, args.bev_proj),
                    os.path.join(vis_dir, f'{safe_sample_name}_bev_uncertainty.png'),
                    f'{sample_name} BEV U_total')
                save_component_heatmaps(
                    sample_name,
                    u_sem_np,
                    u_conf_np,
                    u_occ_np,
                    u_total_np,
                    os.path.join(vis_dir, f'{safe_sample_name}_uncertainty_components.png'),
                    mode=args.bev_proj)
                save_error_overlay(
                    sample_name,
                    u_total_np,
                    error_mask,
                    os.path.join(vis_dir, f'{safe_sample_name}_bev_error_overlay.png'),
                    mode=args.bev_proj)
                save_occlusion_masks(
                    sample_name,
                    mask_camera_np,
                    mask_lidar_np,
                    u_occ_np,
                    os.path.join(vis_dir, f'{safe_sample_name}_occlusion_masks.png'),
                    mode=args.bev_proj)

            per_sample_rows.append(row)
            sample_count += 1
            logger.info('Processed %s (%d valid voxels, error_rate=%.6f)', sample_name, int(valid_mask.sum()), global_error_rate)

    if sample_count == 0:
        raise RuntimeError('No samples were processed. Check dataset split and --max-samples.')

    per_class = []
    for cls_idx in range(num_classes):
        stats = class_acc[cls_idx]
        count = stats['count']
        union = stats['union']
        per_class.append({
            'class_index': cls_idx,
            'class_name': class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            'num_voxels': count,
            'mean_U_total': stats['uncertainty_sum'] / count if count else float('nan'),
            'error_rate': stats['error_sum'] / count if count else float('nan'),
            'IoU': stats['intersection'] / union if union else float('nan'),
        })

    distance_summary = {}
    for name, stats in distance_acc.items():
        count = stats['voxel_count']
        distance_summary[name] = {
            'voxel_count': count,
            'mean_U_total': stats['uncertainty_sum'] / count if count else float('nan'),
            'error_rate': stats['error_sum'] / count if count else float('nan'),
        }

    error_u = mean_from_sum_count(component_acc['mean_U_total_on_error_voxels'])
    correct_u = mean_from_sum_count(component_acc['mean_U_total_on_correct_voxels'])
    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': sample_count,
        'ignore_index': ignore_index,
        'num_classes': num_classes,
        'eval_mask': args.eval_mask,
        'fusion_weights': fusion_weights,
        'error_correlation': {
            'mean_U_total_on_error_voxels': error_u,
            'mean_U_total_on_correct_voxels': correct_u,
            'U_total_error_correct_ratio': error_u / correct_u if correct_u and not math.isnan(correct_u) else float('nan'),
            'mean_U_sem_on_error_voxels': mean_from_sum_count(component_acc['mean_U_sem_on_error_voxels']),
            'mean_U_sem_on_correct_voxels': mean_from_sum_count(component_acc['mean_U_sem_on_correct_voxels']),
            'mean_U_occ_on_error_voxels': mean_from_sum_count(component_acc['mean_U_occ_on_error_voxels']),
            'mean_U_occ_on_correct_voxels': mean_from_sum_count(component_acc['mean_U_occ_on_correct_voxels']),
            'mean_U_conf_on_error_voxels': mean_from_sum_count(component_acc['mean_U_conf_on_error_voxels']),
            'mean_U_conf_on_correct_voxels': mean_from_sum_count(component_acc['mean_U_conf_on_correct_voxels']),
        },
        'topk_error_rates': {
            'error_rate_all_valid_voxels': total_valid_errors / total_valid_voxels if total_valid_voxels else float('nan'),
            'error_rate_top5_uncertainty': topk_error_count['top5'] / topk_voxel_count['top5'] if topk_voxel_count['top5'] else float('nan'),
            'error_rate_top10_uncertainty': topk_error_count['top10'] / topk_voxel_count['top10'] if topk_voxel_count['top10'] else float('nan'),
            'error_rate_top20_uncertainty': topk_error_count['top20'] / topk_voxel_count['top20'] if topk_voxel_count['top20'] else float('nan'),
        },
        'occlusion_regions': finalize_region_stats(occlusion_region_acc),
        'per_class': per_class,
        'per_class_uncertainty_top10': sorted(
            [item for item in per_class if not math.isnan(item['mean_U_total'])],
            key=lambda x: x['mean_U_total'],
            reverse=True)[:10],
        'distance': distance_summary,
    }

    summary_path = os.path.join(args.out_dir, 'summary.json')
    save_json(summary_path, summary)

    csv_path = os.path.join(args.out_dir, 'per_sample_metrics.csv')
    if per_sample_rows:
        fieldnames = sorted({key for row in per_sample_rows for key in row.keys()})
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    logger.info('Saved summary to %s', summary_path)
    logger.info('Saved per-sample metrics to %s', csv_path)
    logger.info('Saved visualizations to %s', vis_dir)
    if args.save_npz:
        logger.info('Saved npz files to %s', npz_dir)
    print_summary(summary)


if __name__ == '__main__':
    main()
