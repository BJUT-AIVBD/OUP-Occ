#!/usr/bin/env python3
"""Shared, inference-only utilities for the Section 4.4 analyses.

This module deliberately lives under ``tools``: it does not register training
components, alter configs/checkpoints, or change the normal model forward path.
"""

import copy
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import mmcv
import mmdet
import numpy as np
import torch
from PIL import Image
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

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
)


CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]
FREE_CLASS = 17
IGNORE_INDEX = 255
EXPECTED_FULL_VAL_SIZE = 6019
DEFAULT_POINT_CLOUD_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
DEFAULT_OCCUPANCY_SIZE = [0.4, 0.4, 0.4]
VISIBILITY_REGIONS = OrderedDict([
    ('visible_by_both', '双模态均可见'),
    ('camera_only', '仅相机可见'),
    ('lidar_only', '仅 LiDAR 可见'),
    ('invisible_by_both', '双模态均不可见'),
])

DEFAULT_BASELINE_CONFIG = (
    'configs/effocc_fusion_r18_data_scales/'
    'flashocc_fusion_r18_base_100%_seqs.py')
DEFAULT_BASELINE_CHECKPOINT = (
    'work_dirs/flashocc_fusion_r18_base_100%_seqs/latest.pth')
DEFAULT_OUP_CONFIG = (
    'configs/oup_occ/oup_occ_fusion_r18_ucrf_udca_100%_seqs.py')
DEFAULT_OUP_CHECKPOINT = (
    'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
    'oup_occ_complete_version_mIoU_54_14.pth')


def absolute_path(path):
    if os.path.isabs(path):
        return os.path.realpath(path)
    return os.path.realpath(os.path.join(ROOT_DIR, path))


def _existing_files(paths):
    return [path for path in paths if path and os.path.isfile(path)]


def resolve_artifact(requested, role, alternatives=None, patterns=None):
    """Resolve a requested file without silently selecting an arbitrary model."""
    alternatives = alternatives or []
    patterns = patterns or []
    requested_abs = absolute_path(requested)
    if os.path.isfile(requested_abs):
        resolved = os.path.realpath(requested_abs)
        print('{}: {} -> {}'.format(role, requested, resolved))
        return resolved

    candidates = _existing_files([absolute_path(path) for path in alternatives])
    if not candidates:
        for pattern in patterns:
            candidates.extend(glob.glob(
                os.path.join(ROOT_DIR, pattern), recursive=True))
        candidates = _existing_files(sorted(set(candidates)))
    if not candidates:
        raise FileNotFoundError(
            '{}不存在，自动核验也未找到候选文件：{}'.format(role, requested_abs))

    # Alternatives are ordered by intended semantic match. Search-only
    # candidates are sorted by mtime merely to make the ambiguity deterministic.
    if alternatives:
        resolved = os.path.realpath(candidates[0])
    else:
        candidates.sort(key=os.path.getmtime, reverse=True)
        resolved = os.path.realpath(candidates[0])
    print('{}默认路径不存在，实际使用: {}'.format(role, resolved))
    if len(candidates) > 1:
        print('  其他候选未使用: {}'.format(
            ', '.join(os.path.realpath(path) for path in candidates[1:6])))
    return resolved


def resolve_oup_paths(config, checkpoint):
    config_path = resolve_artifact(
        config,
        'OUP-Occ config',
        alternatives=[
            'configs/oup_occ/oup_occ_fusion_r18_complete_version_mIoU_54_14.py',
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
            'oup_occ_fusion_r18_ucrf_udca_100%_seqs.py',
            'configs/oup_occ/abandon/'
            'oup_occ_fusion_r18_ucrf_udca_100%_seqs.py',
        ],
        patterns=[
            'configs/oup_occ/**/*complete*54_14*.py',
            'configs/oup_occ/**/*ucrf*udca*.py',
        ])
    checkpoint_path = resolve_artifact(
        checkpoint,
        'OUP-Occ checkpoint',
        alternatives=[DEFAULT_OUP_CHECKPOINT],
        patterns=[
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
            '*complete*54_14*.pth',
        ])
    return config_path, checkpoint_path


def resolve_pair_paths(
        baseline_config, baseline_checkpoint, oup_config, oup_checkpoint):
    baseline_config_path = resolve_artifact(
        baseline_config,
        'Baseline config',
        alternatives=[DEFAULT_BASELINE_CONFIG],
        patterns=['configs/**/*flashocc_fusion_r18_base_100%_seqs.py'])
    baseline_checkpoint_path = resolve_artifact(
        baseline_checkpoint,
        'Baseline checkpoint',
        alternatives=[
            DEFAULT_BASELINE_CHECKPOINT,
            'work_dirs/flashocc_fusion_r18_base_100%_seqs/epoch_45.pth',
        ],
        patterns=[
            'work_dirs/flashocc_fusion_r18_base_100%_seqs/*.pth',
        ])
    oup_config_path, oup_checkpoint_path = resolve_oup_paths(
        oup_config, oup_checkpoint)
    return {
        'baseline_config': baseline_config_path,
        'baseline_checkpoint': baseline_checkpoint_path,
        'oup_config': oup_config_path,
        'oup_checkpoint': oup_checkpoint_path,
    }


def load_cfg(path):
    cfg = compat_cfg(Config.fromfile(path))
    setup_multi_processes(cfg)
    return cfg


def _split_cfg(cfg, split):
    if split not in cfg.data:
        raise KeyError('配置中不存在 data.{}'.format(split))
    return cfg.data[split]


def validate_same_dataset(baseline_cfg, oup_cfg, split):
    baseline_data = _split_cfg(baseline_cfg, split)
    oup_data = _split_cfg(oup_cfg, split)
    for key in ['type', 'ann_file', 'data_root']:
        baseline_value = baseline_data.get(key)
        oup_value = oup_data.get(key)
        if str(baseline_value) != str(oup_value):
            raise ValueError(
                'Baseline/OUP data.{}.{} 不一致: {} vs {}'.format(
                    split, key, baseline_value, oup_value))


def patch_analysis_pipeline(dataset_cfg):
    """Flatten test augmentation and add GT/masks needed by analysis only."""
    dataset_cfg = copy.deepcopy(dataset_cfg)
    flattened = []
    for step in dataset_cfg.pipeline:
        step = copy.deepcopy(step)
        if step.get('type') == 'MultiScaleFlipAug3D':
            flattened.extend(copy.deepcopy(step.get('transforms', [])))
        else:
            flattened.append(step)

    if not any(step.get('type') == 'LoadOccGTFromFile'
               for step in flattened):
        insert_at = 0
        for index, step in enumerate(flattened):
            if step.get('type') == 'ToEgo':
                insert_at = index + 1
                break
            if step.get('type') in ['BEVAug', 'PointToMultiViewDepthFusion']:
                insert_at = index
                break
        flattened.insert(insert_at, dict(type='LoadOccGTFromFile'))

    required = [
        'points', 'img_inputs', 'gt_depth', 'voxel_semantics',
        'mask_lidar', 'mask_camera',
    ]
    has_collect = False
    for step in flattened:
        if step.get('type') == 'PrepareImageInputs':
            step['is_train'] = False
        elif step.get('type') == 'LoadPointsFromMultiSweeps':
            # The original Occ3D test configs omit this flag and therefore
            # randomly select 9 of 10 sweeps in most val samples.
            step['test_mode'] = True
        elif step.get('type') == 'BEVAug':
            step['is_train'] = False
        elif step.get('type') == 'DefaultFormatBundle3D':
            step['with_label'] = False
        elif step.get('type') == 'Collect3D':
            has_collect = True
            keys = list(step.get('keys', []))
            step['keys'] = keys + [key for key in required if key not in keys]
    if not has_collect:
        flattened.append(dict(type='Collect3D', keys=required))
    dataset_cfg.pipeline = flattened
    dataset_cfg.test_mode = True
    return dataset_cfg


def validate_dataset_files(dataset_cfg):
    data_root = dataset_cfg.get('data_root')
    ann_file = dataset_cfg.get('ann_file')
    if data_root and not os.path.isabs(data_root):
        data_root = os.path.join(ROOT_DIR, data_root)
    if ann_file and not os.path.isabs(ann_file):
        ann_file = os.path.join(ROOT_DIR, ann_file)
    if data_root and not os.path.isdir(data_root):
        raise FileNotFoundError('数据集目录不存在: {}'.format(data_root))
    if not ann_file or not os.path.isfile(ann_file):
        raise FileNotFoundError('validation 标注文件不存在: {}'.format(ann_file))
    return os.path.realpath(data_root), os.path.realpath(ann_file)


def build_analysis_dataset(cfg, split):
    dataset_cfg = patch_analysis_pipeline(_split_cfg(cfg, split))
    data_root, ann_file = validate_dataset_files(dataset_cfg)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) <= 0:
        raise RuntimeError('构建的数据集为空')
    first_info = dataset.data_infos[0]
    occ_path = first_info.get('occ_path')
    labels_path = (
        occ_path if str(occ_path).endswith('labels.npz')
        else os.path.join(str(occ_path), 'labels.npz'))
    if not os.path.isfile(labels_path):
        raise FileNotFoundError('首个 Occ3D GT 不存在: {}'.format(labels_path))
    return dataset_cfg, dataset, data_root, ann_file


def build_analysis_loader(dataset, workers_per_gpu, seed):
    return build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
        seed=seed)


def disable_pretrained(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg['pretrained'] = None
    if isinstance(model_cfg.get('img_backbone'), dict):
        model_cfg.img_backbone.pretrained = None
    model_cfg['train_cfg'] = None
    return model_cfg


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith('module.') for key in state_dict):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def inspect_checkpoint(model, checkpoint_path):
    """Reject shape-incompatible or parameter-incomplete checkpoints."""
    raw = torch.load(checkpoint_path, map_location='cpu')
    state_dict = raw.get('state_dict', raw)
    state_dict = _strip_module_prefix(state_dict)
    model_state = model.state_dict()
    model_parameters = set(dict(model.named_parameters()).keys())
    shape_mismatches = []
    for key in sorted(set(model_state) & set(state_dict)):
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
            shape_mismatches.append({
                'key': key,
                'model_shape': list(model_state[key].shape),
                'checkpoint_shape': list(state_dict[key].shape),
            })
    missing_parameters = sorted(model_parameters - set(state_dict))
    missing_state = sorted(set(model_state) - set(state_dict))
    unexpected_state = sorted(set(state_dict) - set(model_state))
    if shape_mismatches or missing_parameters:
        raise RuntimeError(
            'checkpoint 与模型不兼容：shape_mismatches={}，'
            'missing_parameters={}。拒绝静默加载错误权重。'.format(
                shape_mismatches[:10], missing_parameters[:20]))
    metadata = raw.get('meta', {}) if isinstance(raw, dict) else {}
    result = {
        'checkpoint': os.path.realpath(checkpoint_path),
        'epoch_meta': metadata.get('epoch'),
        'model_state_keys': len(model_state),
        'checkpoint_state_keys': len(state_dict),
        'matched_state_keys': len(set(model_state) & set(state_dict)),
        'missing_state_keys': missing_state,
        'unexpected_state_keys': unexpected_state,
        'shape_mismatches': shape_mismatches,
        'all_model_parameters_present': not missing_parameters,
    }
    del raw, state_dict
    return result


def build_eval_model(cfg, checkpoint_path, dataset, gpu_id, role):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('{} checkpoint 不存在: {}'.format(
            role, checkpoint_path))
    model = build_model(
        disable_pretrained(cfg.model), test_cfg=cfg.get('test_cfg'))
    checkpoint_audit = inspect_checkpoint(model, checkpoint_path)
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location='cpu', strict=False)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    model = MMDataParallel(
        model.float().cuda(gpu_id), device_ids=[gpu_id])
    model.eval()
    print(
        '{}: class={}，checkpoint epoch_meta={}，'
        '参数完整={}，matched_keys={}/{}'.format(
            role,
            model.module.__class__.__name__,
            checkpoint_audit['epoch_meta'],
            checkpoint_audit['all_model_parameters_present'],
            checkpoint_audit['matched_state_keys'],
            checkpoint_audit['model_state_keys']))
    return model, checkpoint_audit


def configure_runtime(seed, gpu_id):
    if not torch.cuda.is_available():
        raise RuntimeError('这些评测工具需要单张 CUDA GPU')
    torch.cuda.set_device(gpu_id)
    set_random_seed(seed, deterministic=True)
    torch.backends.cudnn.benchmark = False


def get_num_classes(cfg):
    for head_name in ['final_occ_head', 'occ_head', 'coarse_occ_head']:
        head = cfg.model.get(head_name)
        if head and head.get('num_classes') is not None:
            return int(head.num_classes)
    raise ValueError('无法从 occupancy head 推断类别数')


def get_ignore_index(cfg):
    for head_name in ['final_occ_head', 'occ_head', 'coarse_occ_head']:
        head = cfg.model.get(head_name)
        if head and head.get('loss_occ'):
            return int(head.loss_occ.get('ignore_index', IGNORE_INDEX))
    return IGNORE_INDEX


def get_class_names(num_classes):
    evaluator_names = Metric_mIoU(num_classes=num_classes).class_names
    if len(evaluator_names) < num_classes:
        raise ValueError('官方 evaluator 类别名数量不足')
    return list(evaluator_names[:num_classes])


def official_protocol(cfg, dataset, official_mask):
    num_classes = get_num_classes(cfg)
    ignore_index = get_ignore_index(cfg)
    names = get_class_names(num_classes)
    metric = Metric_mIoU(
        num_classes=num_classes,
        use_lidar_mask=official_mask == 'lidar',
        use_image_mask=official_mask == 'camera')
    protocol = {
        'dataset': 'Occ3D-nuScenes validation',
        'dataset_samples': len(dataset),
        'expected_full_val_samples': EXPECTED_FULL_VAL_SIZE,
        'official_mask_argument': official_mask,
        'repository_official_evaluator_mask': 'mask_camera',
        'mask_tensor_key': (
            'mask_camera' if official_mask == 'camera'
            else 'mask_lidar' if official_mask == 'lidar'
            else official_mask),
        'num_classes': num_classes,
        'class_names': names,
        'ignore_index': ignore_index,
        'free_class_index': num_classes - 1,
        'miou_class_indices': list(range(num_classes - 1)),
        'miou_iou_output_scale': 'percent (0-100)',
        'error_rate_output_scale': 'ratio (0-1)',
        'miou_absent_class_handling': (
            'union=0 时 IoU=NaN，并由 numpy.nanmean 排除'),
        'point_cloud_range': list(metric.point_cloud_range),
        'occupancy_size': list(metric.occupancy_size),
        'grid_shape': [
            metric.occ_xdim, metric.occ_ydim, metric.occ_zdim],
        'model_mode': 'model.eval()',
        'gradient_mode': 'torch.no_grad()',
        'batch_size': 1,
        'gpu_count': 1,
        'test_sweep_selection': (
            'LoadPointsFromMultiSweeps.test_mode=True，固定最近 sweeps'),
    }
    if official_mask != 'camera':
        protocol['protocol_warning'] = (
            '仓库 NuScenesDatasetOccpancy.evaluate 固定使用 mask_camera；'
            '当前参数不是仓库官方默认口径。')
    return protocol


def print_protocol(paths, protocol):
    print('\n===== 执行前协议核验 =====')
    for key, value in paths.items():
        print('{}: {}'.format(key, value))
    print('数据集样本数: {}'.format(protocol['dataset_samples']))
    print('评价 mask: {}（仓库官方: {}）'.format(
        protocol['mask_tensor_key'],
        protocol['repository_official_evaluator_mask']))
    print('类别数: {}'.format(protocol['num_classes']))
    print('ignore index: {}'.format(protocol['ignore_index']))
    print('batch size / GPU: 1 / 1')
    print('==========================\n')


def sample_limit(args, dataset_size):
    if getattr(args, 'full_val', False):
        return dataset_size
    max_samples = getattr(args, 'max_samples', None)
    if max_samples is None or max_samples < 0:
        return dataset_size
    if max_samples == 0:
        raise ValueError('--max-samples 必须大于 0，或使用 --full-val')
    return min(dataset_size, max_samples)


def unwrap_data(value):
    if isinstance(value, DC):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_data(value[0])
    return value


def require_batch_tensor(data_batch, key):
    if key not in data_batch:
        raise KeyError('{} 不在 data_batch 中'.format(key))
    value = unwrap_data(data_batch[key])
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError('{} 应为 Tensor，实际为 {}'.format(key, type(value)))
    return value


def get_img_metas(data_batch):
    metas = unwrap_data(data_batch.get('img_metas'))
    while (isinstance(metas, (list, tuple)) and len(metas) == 1
           and not isinstance(metas[0], dict)):
        metas = metas[0]
    if isinstance(metas, dict):
        return [metas]
    if isinstance(metas, (list, tuple)):
        return list(metas)
    return []


def sample_token(data_batch, fallback_index):
    metas = get_img_metas(data_batch)
    meta = metas[0] if metas else {}
    for key in ['sample_token', 'token', 'sample_idx']:
        if meta.get(key) is not None:
            return str(meta[key])
    return 'sample_{:06d}'.format(fallback_index)


def extract_logits(output, prefer='final'):
    while isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if not isinstance(output, dict):
        return output
    if prefer == 'coarse':
        keys = ['coarse_occ_logits']
    else:
        keys = [
            'output_occ_logits', 'compensated_occ_logits',
            'final_occ_logits',
        ]
    for key in keys:
        if key in output:
            return output[key]
    raise KeyError('模型输出中不存在 {} occupancy logits'.format(prefer))


def predict_from_output(output, num_classes, prefer='final'):
    logits = align_logits_to_bxyzc(
        extract_logits(output, prefer=prefer),
        num_classes=num_classes)
    return logits.argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)


def infer_model(model, data_batch):
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        return model(return_loss=True, return_result=True, **data_batch)


def output_structure(value, depth=0, max_depth=3):
    if isinstance(value, torch.Tensor):
        return {
            'type': 'Tensor',
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'device': str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            'type': 'ndarray',
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key): output_structure(item, depth + 1, max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            output_structure(item, depth + 1, max_depth)
            for item in value[:5]
        ]
    return type(value).__name__


def extract_gt_masks(data_batch, spatial_shape, num_classes, ignore_index,
                     official_mask):
    gt = align_tensor_to_bxyz(
        require_batch_tensor(data_batch, 'voxel_semantics'),
        target_shape=spatial_shape,
        name='voxel_semantics')[0].detach().cpu().numpy().astype(np.int16)
    mask_camera = align_tensor_to_bxyz(
        require_batch_tensor(data_batch, 'mask_camera'),
        target_shape=spatial_shape,
        name='mask_camera')[0].detach().cpu().numpy().astype(bool)
    mask_lidar = align_tensor_to_bxyz(
        require_batch_tensor(data_batch, 'mask_lidar'),
        target_shape=spatial_shape,
        name='mask_lidar')[0].detach().cpu().numpy().astype(bool)
    label_valid = (
        (gt != ignore_index) & (gt >= 0) & (gt < num_classes))
    if official_mask == 'camera':
        eval_mask = mask_camera
    elif official_mask == 'lidar':
        eval_mask = mask_lidar
    elif official_mask == 'both':
        eval_mask = mask_camera & mask_lidar
    elif official_mask == 'union':
        eval_mask = mask_camera | mask_lidar
    elif official_mask == 'all':
        eval_mask = np.ones_like(label_valid, dtype=bool)
    else:
        raise ValueError('未知 official mask: {}'.format(official_mask))
    return gt, mask_camera, mask_lidar, label_valid & eval_mask, label_valid


def visibility_masks(mask_camera, mask_lidar):
    return OrderedDict([
        ('visible_by_both', mask_camera & mask_lidar),
        ('camera_only', mask_camera & ~mask_lidar),
        ('lidar_only', ~mask_camera & mask_lidar),
        ('invisible_by_both', ~mask_camera & ~mask_lidar),
    ])


def confusion_matrix(pred, gt, valid, num_classes):
    pred = np.asarray(pred)[valid].astype(np.int64)
    gt = np.asarray(gt)[valid].astype(np.int64)
    keep = (
        (pred >= 0) & (pred < num_classes)
        & (gt >= 0) & (gt < num_classes))
    encoded = num_classes * gt[keep] + pred[keep]
    return np.bincount(
        encoded, minlength=num_classes ** 2).reshape(
            num_classes, num_classes)


def per_class_iou(hist):
    hist = np.asarray(hist, dtype=np.float64)
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(
        np.diag(hist),
        denominator,
        out=np.full(hist.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0)


def metrics_from_hist(hist, free_class):
    ious = per_class_iou(hist)
    included = ious[:free_class]
    miou = (
        float(np.nanmean(included)) * 100.0
        if np.isfinite(included).any() else float('nan'))
    return {
        # Match NuScenesDatasetOccpancy.evaluate/Metric_mIoU reporting scale.
        'miou': miou,
        'per_class_iou': ious * 100.0,
        'scale': 'percent',
    }


def distance_volume(shape, point_cloud_range):
    """Compute horizontal distance from physical voxel-center coordinates."""
    xdim, ydim, zdim = [int(value) for value in shape]
    x_min, y_min, _, x_max, y_max, _ = [
        float(value) for value in point_cloud_range]
    x_size = (x_max - x_min) / xdim
    y_size = (y_max - y_min) / ydim
    xs = x_min + (np.arange(xdim, dtype=np.float64) + 0.5) * x_size
    ys = y_min + (np.arange(ydim, dtype=np.float64) + 0.5) * y_size
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    distance = np.sqrt(xx ** 2 + yy ** 2)
    return np.broadcast_to(distance[:, :, None], (xdim, ydim, zdim))


def parse_distance_edges(values):
    parsed = []
    for value in values:
        if str(value).lower() in ['inf', '+inf', 'infinity']:
            parsed.append(float('inf'))
        else:
            parsed.append(float(value))
    if len(parsed) < 2:
        raise ValueError('--distance-bins 至少需要两个边界')
    if parsed[0] < 0 or any(
            right <= left for left, right in zip(parsed[:-1], parsed[1:])):
        raise ValueError('--distance-bins 必须严格递增且从非负数开始')
    return parsed


def distance_bin_specs(edges):
    specs = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if math.isinf(upper):
            name = '{:g}m+'.format(lower)
        else:
            name = '{:g}-{:g}m'.format(lower, upper)
        specs.append((name, lower, upper))
    return specs


def safe_divide(numerator, denominator):
    if denominator == 0:
        return float('nan')
    return float(numerator) / float(denominator)


def finite_or_none(value):
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(
            json_safe(value),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    'NaN' if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value)
                for key, value in row.items()
            })


def fmt(value, digits=4, percent=False):
    if value is None:
        return 'NaN'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return 'NaN'
    if percent:
        value *= 100.0
    return ('{:.%df}' % digits).format(value)


def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def finish_message(name, out_dir, processed, dataset_size, elapsed):
    print('\n===== {} 核心结果已生成 ====='.format(name))
    print('结果目录: {}'.format(out_dir))
    print('实际样本数: {} / {}'.format(processed, dataset_size))
    print('运行时长: {:.2f} 分钟'.format(elapsed / 60.0))
    print('完整覆盖 6019 个验证样本: {}'.format(
        processed == dataset_size == EXPECTED_FULL_VAL_SIZE))


def stable_rng(seed, condition, token, unit='sample'):
    material = '{}|{}|{}|{}'.format(
        int(seed), condition, token, unit).encode('utf-8')
    seed64 = int.from_bytes(
        hashlib.sha256(material).digest()[:8], byteorder='little')
    return np.random.default_rng(seed64)


def tensor_sha256(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('utf-8'))
    digest.update(str(array.shape).encode('utf-8'))
    digest.update(array.tobytes())
    return digest.hexdigest()


CONDITION_SPECS = OrderedDict([
    ('clean', dict(point_drop=0.0, camera_drop=0)),
    ('lidar_drop25', dict(point_drop=0.25, camera_drop=0)),
    ('lidar_drop50', dict(point_drop=0.50, camera_drop=0)),
    ('camera_drop1', dict(point_drop=0.0, camera_drop=1)),
    ('camera_drop3', dict(point_drop=0.0, camera_drop=3)),
    ('brightness_0.5', dict(
        point_drop=0.0, camera_drop=0, brightness=0.5)),
    ('gaussian_blur', dict(
        point_drop=0.0, camera_drop=0,
        gaussian_blur=dict(kernel_size=7, sigma=1.5))),
    ('combined', dict(point_drop=0.50, camera_drop=1)),
])


class CorruptedImagePreparation:
    """Apply image corruption to raw pixels immediately before Normalize."""

    def __init__(self, transform, condition, seed):
        self.transform = transform
        self.condition = condition
        self.seed = int(seed)
        self.spec = CONDITION_SPECS[condition]

    def __call__(self, results):
        token = str(results.get('sample_idx', results.get('token', 'unknown')))
        cam_names = list(self.transform.data_config['cams'])
        camera_drop = int(self.spec.get('camera_drop', 0))
        rng = stable_rng(
            self.seed, self.condition, token, unit='failed_cameras')
        if camera_drop:
            failed = sorted(rng.choice(
                cam_names, size=camera_drop, replace=False).tolist())
        else:
            failed = []

        n_frames = 1 + (
            len(results.get('adjacent', []))
            if self.transform.sequential else 0)
        original_normalize = self.transform.normalize_img
        image_records = []
        call_index = {'value': 0}

        def corrupt_then_normalize(image):
            index = call_index['value']
            call_index['value'] += 1
            cam_index = min(index // max(n_frames, 1), len(cam_names) - 1)
            camera_name = cam_names[cam_index]
            frame_index = index % max(n_frames, 1)
            array = np.asarray(image).astype(np.float32)
            original_mean = float(array.mean())

            if camera_name in failed:
                corrupted = np.zeros_like(array)
            elif self.spec.get('brightness') is not None:
                corrupted = np.clip(
                    array * float(self.spec['brightness']), 0.0, 255.0)
            elif self.spec.get('gaussian_blur') is not None:
                blur = self.spec['gaussian_blur']
                kernel = int(blur['kernel_size'])
                corrupted = cv2.GaussianBlur(
                    array, (kernel, kernel), float(blur['sigma']))
            else:
                corrupted = array

            corrupted = np.clip(corrupted, 0.0, 255.0).astype(np.uint8)
            image_records.append({
                'camera': camera_name,
                'frame_index': frame_index,
                'original_mean': original_mean,
                'corrupted_mean': float(corrupted.mean()),
            })
            return original_normalize(Image.fromarray(corrupted))

        self.transform.normalize_img = corrupt_then_normalize
        try:
            results = self.transform(results)
        finally:
            self.transform.normalize_img = original_normalize
        failed_frames_are_zero = all(
            record['corrupted_mean'] == 0.0
            for record in image_records
            if record['camera'] in failed)
        failed_frame_counts = {
            camera: sum(
                record['camera'] == camera for record in image_records)
            for camera in failed
        }
        results['degradation_audit'] = {
            'condition': self.condition,
            'sample_token': token,
            'failed_cameras': failed,
            'camera_frames_consistent': (
                failed_frames_are_zero
                and all(
                    count == n_frames
                    for count in failed_frame_counts.values())),
            'failed_camera_frame_counts': failed_frame_counts,
            'image_stage': (
                '几何 resize/crop 后的原始像素，mmlabNormalize 之前'),
            'image_records': image_records,
            'explicit_per_camera_availability_mask_present': False,
            'voxel_visibility_masks_modified': False,
        }
        return results


class DeterministicPointDropout:
    """Drop merged keyframe+sweep points before depth projection/voxelization."""

    def __init__(self, condition, seed):
        self.condition = condition
        self.seed = int(seed)
        self.spec = CONDITION_SPECS[condition]

    def __call__(self, results):
        points = results['points']
        original_count = int(points.tensor.shape[0])
        drop_fraction = float(self.spec.get('point_drop', 0.0))
        keep_count = int(round(original_count * (1.0 - drop_fraction)))
        if original_count > 0:
            keep_count = max(1, min(original_count, keep_count))
        if keep_count < original_count:
            token = str(results.get(
                'sample_idx', results.get('token', 'unknown')))
            rng = stable_rng(
                self.seed, self.condition, token, unit='lidar_points')
            indices = np.sort(rng.choice(
                original_count, size=keep_count, replace=False))
            points = points[indices]
            results['points'] = points
        audit = results.setdefault('degradation_audit', {})
        audit.update({
            'point_stage': (
                '多 sweep 合并、ToEgo/BEVAug 后，'
                'PointToMultiViewDepthFusion 与 voxelization 前'),
            'point_drop_fraction_requested': drop_fraction,
            'original_points': original_count,
            'remaining_points': int(results['points'].tensor.shape[0]),
            'dropped_points': (
                original_count - int(results['points'].tensor.shape[0])),
        })
        return results


def install_sensor_corruption(dataset, condition, seed):
    """Mutate only this in-memory test dataset pipeline."""
    if condition not in CONDITION_SPECS:
        raise ValueError('不支持的退化条件: {}'.format(condition))
    transforms = dataset.pipeline.transforms
    prepare_index = None
    depth_index = None
    bev_index = None
    collect_transform = None
    for index, transform in enumerate(transforms):
        name = transform.__class__.__name__
        if name == 'PrepareImageInputs':
            prepare_index = index
        elif name == 'LoadPointsFromMultiSweeps':
            transform.test_mode = True
        elif name == 'BEVAug':
            bev_index = index
        elif name == 'PointToMultiViewDepthFusion':
            depth_index = index
        elif name == 'Collect3D':
            collect_transform = transform
    if prepare_index is None or depth_index is None or bev_index is None:
        raise RuntimeError(
            '无法定位 PrepareImageInputs/BEVAug/'
            'PointToMultiViewDepthFusion 注入点')
    transforms[prepare_index] = CorruptedImagePreparation(
        transforms[prepare_index], condition, seed)
    # Insert after BEVAug and before depth generation.
    if not (bev_index < depth_index):
        raise RuntimeError('BEVAug 必须位于深度生成之前')
    transforms.insert(
        depth_index, DeterministicPointDropout(condition, seed))
    if collect_transform is None:
        raise RuntimeError('无法定位 Collect3D 以保存退化核验信息')
    if 'degradation_audit' not in collect_transform.meta_keys:
        collect_transform.meta_keys = tuple(
            collect_transform.meta_keys) + ('degradation_audit',)
    return CONDITION_SPECS[condition]


def batch_sensor_hashes(data_batch):
    points = unwrap_data(data_batch.get('points'))
    if hasattr(points, 'tensor'):
        points = points.tensor
    images = unwrap_data(data_batch.get('img_inputs'))
    while isinstance(images, (list, tuple)) and len(images) == 1:
        images = images[0]
    image_tensor = images[0] if isinstance(images, (list, tuple)) else images
    depth = require_batch_tensor(data_batch, 'gt_depth')
    return {
        'points_sha256': tensor_sha256(points),
        'images_sha256': tensor_sha256(image_tensor),
        'gt_depth_sha256': tensor_sha256(depth),
    }


def degradation_audit_from_batch(data_batch):
    metas = get_img_metas(data_batch)
    if not metas:
        return {}
    return copy.deepcopy(metas[0].get('degradation_audit', {}))


class RuntimeLogger:
    def __init__(self, path):
        self.path = path

    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = '{} - {}'.format(timestamp, message)
        print(line)
        with open(self.path, 'a', encoding='utf-8') as file:
            file.write(line + '\n')
