#!/usr/bin/env python3
"""Select and export qualitative Occ3D Baseline/OUP-Occ cases.

This script performs inference only. It never trains a model and never writes
to an existing config or checkpoint. All metrics and selection reasons are
computed from actual predictions on the official ``mask_camera`` region.
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import sys
import time
import warnings

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mmcv  # noqa: E402
import mmdet  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import DataContainer as DC  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet.apis import set_random_seed  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.models.occ_uncertainty import (  # noqa: E402
    align_logits_to_bxyzc,
    align_tensor_to_bxyz,
)


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
DEFAULT_OUTPUT_DIR = 'work_dirs/qualitative_occ3d'

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
FREE_CLASS = 17
IGNORE_INDEX = 255
POINT_CLOUD_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
BEV_EXTENT = [
    POINT_CLOUD_RANGE[0],
    POINT_CLOUD_RANGE[3],
    POINT_CLOUD_RANGE[1],
    POINT_CLOUD_RANGE[4],
]

# Occ3D colors used by the repository's official visualization script.
CLASS_COLORS = np.asarray([
    [0, 0, 0],
    [112, 128, 144],
    [220, 20, 60],
    [255, 127, 80],
    [255, 158, 0],
    [233, 150, 70],
    [255, 61, 99],
    [0, 0, 230],
    [47, 79, 79],
    [255, 140, 0],
    [255, 99, 71],
    [0, 207, 191],
    [175, 0, 75],
    [75, 0, 75],
    [112, 180, 60],
    [222, 184, 135],
    [0, 175, 0],
    [255, 255, 255],
], dtype=np.uint8)

CAMERA_FILES = [
    ('CAM_FRONT', 'cam_front.png'),
    ('CAM_FRONT_LEFT', 'cam_front_left.png'),
    ('CAM_FRONT_RIGHT', 'cam_front_right.png'),
    ('CAM_BACK', 'cam_back.png'),
    ('CAM_BACK_LEFT', 'cam_back_left.png'),
    ('CAM_BACK_RIGHT', 'cam_back_right.png'),
]

VEHICLE_CLASSES = {2, 3, 4, 5, 6, 9, 10}
SMALL_OBJECT_CLASSES = {2, 6, 7, 8}
ROAD_CLASSES = {1, 11, 12, 13, 14}
DISTANCE_BINS = [
    ('0-15m', 0.0, 15.0),
    ('15-30m', 15.0, 30.0),
    ('30-45m', 30.0, 45.0),
    ('45m+', 45.0, float('inf')),
]

REQUESTED_INTERMEDIATES = [
    'reliability_camera',
    'reliability_lidar',
    'cross_modal_discrepancy',
    'uncertainty_semantic',
    'uncertainty_confidence',
    'uncertainty_occlusion',
    'uncertainty_bev',
    'query_locations',
    'camera_attention',
    'lidar_attention',
    'sampling_offsets',
    'refinement_gate',
    'stage1_prediction',
    'final_prediction',
]
_DISTANCE_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='在完整 Occ3D-nuScenes val 集上筛选并导出定性案例')
    parser.add_argument(
        '--baseline-config', default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument(
        '--baseline-checkpoint', default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument('--oup-config', default=DEFAULT_OUP_CONFIG)
    parser.add_argument('--oup-checkpoint', default=DEFAULT_OUP_CHECKPOINT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--success-count', type=int, default=6)
    parser.add_argument('--failure-count', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='仅用于调试；默认 -1 表示完整 val 集')
    parser.add_argument(
        '--max-3d-points',
        type=int,
        default=80000,
        help='单张 3D 预测图的确定性可视化采样上限')
    return parser.parse_args()


def absolute_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def _existing(paths):
    return [path for path in paths if path and os.path.isfile(path)]


def resolve_artifact(requested, role, alternatives, search_patterns):
    requested_abs = absolute_path(requested)
    direct = _existing([requested_abs])
    if direct:
        resolved = os.path.realpath(direct[0])
        print('{}: {} -> {}'.format(role, requested, resolved))
        return resolved

    alternative_paths = _existing([absolute_path(path) for path in alternatives])
    if alternative_paths:
        resolved = os.path.realpath(alternative_paths[0])
        print('{}默认路径不存在，自动确认实际文件: {}'.format(role, resolved))
        return resolved

    matches = []
    for pattern in search_patterns:
        matches.extend(glob.glob(absolute_path(pattern), recursive=True))
    matches = _existing(sorted(set(matches)))
    if not matches:
        raise FileNotFoundError(
            '{}不存在，且自动搜索未找到候选文件。请求路径: {}'.format(
                role, requested_abs))
    matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    resolved = os.path.realpath(matches[0])
    print('{}默认路径不存在，自动搜索并确认: {}'.format(role, resolved))
    if len(matches) > 1:
        print('  其余候选: {}'.format(', '.join(matches[1:6])))
    return resolved


def resolve_all_paths(args):
    baseline_config = resolve_artifact(
        args.baseline_config,
        'Baseline config',
        [DEFAULT_BASELINE_CONFIG],
        ['configs/**/*flashocc_fusion_r18_base_100%_seqs.py'])
    baseline_checkpoint = resolve_artifact(
        args.baseline_checkpoint,
        'Baseline checkpoint',
        [
            'ckpts/effocc_fusion_r18.pth',
            'work_dirs/flashocc_fusion_r18_base_100%_seqs/latest.pth',
        ],
        [
            'work_dirs/flashocc_fusion_r18_base_100%_seqs/*.pth',
            'ckpts/*effocc*fusion*r18*.pth',
        ])
    oup_config = resolve_artifact(
        args.oup_config,
        'OUP-Occ config',
        [
            'configs/oup_occ/oup_occ_fusion_r18_complete_version_mIoU_54_14.py',
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
            'oup_occ_fusion_r18_ucrf_udca_100%_seqs.py',
            'configs/oup_occ/abandon/'
            'oup_occ_fusion_r18_ucrf_udca_100%_seqs.py',
        ],
        [
            'configs/oup_occ/**/*ucrf*udca*.py',
            'configs/oup_occ/**/*complete*54_14*.py',
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/*.py',
        ])
    oup_checkpoint = resolve_artifact(
        args.oup_checkpoint,
        'OUP-Occ checkpoint',
        [DEFAULT_OUP_CHECKPOINT],
        [
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
            '*54_14*.pth',
            'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/*.pth',
        ])
    return {
        'baseline_config': baseline_config,
        'baseline_checkpoint': baseline_checkpoint,
        'oup_config': oup_config,
        'oup_checkpoint': oup_checkpoint,
    }


def patch_analysis_pipeline(dataset_cfg):
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
        'mask_lidar', 'mask_camera'
    ]
    has_collect = False
    for step in flattened:
        if step.get('type') == 'PrepareImageInputs':
            step['is_train'] = False
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


def load_cfg(path):
    cfg = compat_cfg(Config.fromfile(path))
    setup_multi_processes(cfg)
    return cfg


def disable_pretrained(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg['pretrained'] = None
    if isinstance(model_cfg.get('img_backbone'), dict):
        model_cfg.img_backbone.pretrained = None
    model_cfg['train_cfg'] = None
    return model_cfg


def build_eval_model(cfg, checkpoint_path, dataset, gpu_id, role):
    model = build_model(
        disable_pretrained(cfg.model),
        test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    model = MMDataParallel(model.float().cuda(gpu_id), device_ids=[gpu_id])
    model.eval()
    print('{} 模型已加载并设为 eval/FP32: {}'.format(
        role, model.module.__class__.__name__))
    return model


def unwrap_data(value):
    if isinstance(value, DC):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_data(value[0])
    return value


def require_batch_tensor(data_batch, key):
    if key not in data_batch:
        raise KeyError('{} 不在 data_batch 中，请检查分析 pipeline'.format(key))
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


def get_points(data_batch):
    value = unwrap_data(data_batch.get('points'))
    if hasattr(value, 'tensor'):
        value = value.tensor
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError('无法从 data_batch 提取点云，实际类型为 {}'.format(type(value)))


def extract_logits(output, prefer_final=True):
    while isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, dict):
        keys = [
            'output_occ_logits', 'compensated_occ_logits',
            'final_occ_logits'
        ]
        if not prefer_final:
            keys = ['coarse_occ_logits'] + keys
        for key in keys:
            if key in output:
                return output[key]
        raise KeyError('模型输出字典中不存在 occupancy logits')
    return output


def predict_from_output(output):
    logits = align_logits_to_bxyzc(
        extract_logits(output, prefer_final=True),
        num_classes=len(CLASS_NAMES))
    return logits.argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)


def coarse_predict_from_output(output):
    if not isinstance(output, dict) or 'coarse_occ_logits' not in output:
        return None
    logits = align_logits_to_bxyzc(
        output['coarse_occ_logits'], num_classes=len(CLASS_NAMES))
    return logits.argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)


def infer_pair(baseline_model, oup_model, data_batch):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        baseline_output = baseline_model(
            return_loss=True, return_result=True, **data_batch)
        baseline_pred = predict_from_output(baseline_output)
        del baseline_output

        oup_output = oup_model(
            return_loss=True, return_result=True, **data_batch)
        oup_pred = predict_from_output(oup_output)
    return baseline_pred, oup_pred, oup_output


def safe_divide(numerator, denominator):
    if denominator == 0:
        return float('nan')
    return float(numerator) / float(denominator)


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def confusion_matrix(pred, gt, valid, num_classes=18):
    pred = pred[valid].astype(np.int64)
    gt = gt[valid].astype(np.int64)
    keep = (
        (pred >= 0) & (pred < num_classes)
        & (gt >= 0) & (gt < num_classes))
    encoded = num_classes * gt[keep] + pred[keep]
    return np.bincount(
        encoded,
        minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_iou(hist):
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(
        np.diag(hist),
        denominator,
        out=np.full(hist.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0)


def occupancy_iou(pred, gt, valid):
    pred_occ = pred != FREE_CLASS
    gt_occ = gt != FREE_CLASS
    intersection = int((pred_occ & gt_occ & valid).sum())
    union = int(((pred_occ | gt_occ) & valid).sum())
    return safe_divide(intersection, union) * 100.0


def distance_volume(shape):
    shape = tuple(shape)
    if shape in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[shape]
    xdim, ydim, zdim = shape
    x_min, y_min, _, x_max, y_max, _ = POINT_CLOUD_RANGE
    xs = x_min + (np.arange(xdim) + 0.5) * (x_max - x_min) / xdim
    ys = y_min + (np.arange(ydim) + 0.5) * (y_max - y_min) / ydim
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    result = np.repeat(
        np.sqrt(xx ** 2 + yy ** 2)[:, :, None], zdim, axis=2)
    _DISTANCE_CACHE[shape] = result
    return result


def road_boundary_mask(gt, valid):
    road = np.isin(gt, list(ROAD_CLASSES)) & valid
    boundary = np.zeros_like(road, dtype=bool)
    for axis in [0, 1]:
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        different = gt[left] != gt[right]
        relevant = (road[left] | road[right]) & different
        boundary[left] |= relevant
        boundary[right] |= relevant
    return boundary & valid


def point_density_statistics(points, gt, valid):
    points = np.asarray(points)
    in_xy = (
        (points[:, 0] >= POINT_CLOUD_RANGE[0])
        & (points[:, 0] < POINT_CLOUD_RANGE[3])
        & (points[:, 1] >= POINT_CLOUD_RANGE[1])
        & (points[:, 1] < POINT_CLOUD_RANGE[4]))
    in_points = points[in_xy]
    xdim, ydim = gt.shape[:2]
    # Compute in float64 and clamp the upper boundary. Float32 points just
    # below 40 m can otherwise round to index 200, making bincount longer than
    # the expected 200*200 grid.
    xy = in_points[:, :2].astype(np.float64, copy=False)
    x_idx = np.floor(
        (xy[:, 0] - POINT_CLOUD_RANGE[0])
        / (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) * xdim).astype(int)
    y_idx = np.floor(
        (xy[:, 1] - POINT_CLOUD_RANGE[1])
        / (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) * ydim).astype(int)
    x_idx = np.clip(x_idx, 0, xdim - 1)
    y_idx = np.clip(y_idx, 0, ydim - 1)
    flat = x_idx * ydim + y_idx
    column_counts = np.bincount(
        flat, minlength=xdim * ydim).reshape(xdim, ydim)
    sparse_columns = column_counts <= 1
    occupied_visible = valid & (gt != FREE_CLASS)
    occupied_columns = occupied_visible.max(axis=2)
    occupied_column_count = int(occupied_columns.sum())
    return {
        'total_point_count': int(points.shape[0]),
        'in_range_point_count': int(in_points.shape[0]),
        'far_point_ratio': safe_divide(
            int((np.linalg.norm(in_points[:, :2], axis=1) >= 30.0).sum()),
            in_points.shape[0]),
        'points_per_visible_occupied_column': safe_divide(
            int(in_points.shape[0]), occupied_column_count),
        'sparse_column_ratio_on_visible_occupancy': safe_divide(
            int((sparse_columns & occupied_columns).sum()),
            occupied_column_count),
        'sparse_columns': sparse_columns,
    }


def tensor_to_bxyz_numpy(value, spatial_shape=None):
    if value is None:
        return None
    value = align_tensor_to_bxyz(
        value,
        target_shape=spatial_shape,
        name='qualitative_intermediate')
    return value.detach().float().cpu().numpy()


def get_occlusion_uncertainty(oup_output, gt_shape):
    if not isinstance(oup_output, dict):
        return None
    uncertainty = oup_output.get('uncertainty_dict', {})
    value = uncertainty.get('U_occ_voxel')
    if value is None:
        return None
    return tensor_to_bxyz_numpy(value, gt_shape)


class NuScenesAnnotationEvidence:
    def __init__(self, data_root):
        self.nusc = None
        self.error = None
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(
                version='v1.0-trainval',
                dataroot=absolute_path(data_root),
                verbose=False)
        except Exception as error:
            self.error = repr(error)
            warnings.warn(
                'nuScenes 标注可见度不可用，将不把样本描述为遮挡车辆: {}'
                .format(self.error))

    def describe(self, sample_token):
        result = {
            'low_visibility_annotation_count': 0,
            'low_visibility_vehicle_count': 0,
            'low_visibility_vehicle_categories': [],
            'annotation_evidence_available': self.nusc is not None,
        }
        if self.nusc is None:
            return result
        try:
            sample = self.nusc.get('sample', sample_token)
            low_vehicle_categories = []
            low_count = 0
            for annotation_token in sample['anns']:
                annotation = self.nusc.get(
                    'sample_annotation', annotation_token)
                if str(annotation.get('visibility_token')) not in ['1', '2']:
                    continue
                low_count += 1
                instance = self.nusc.get(
                    'instance', annotation['instance_token'])
                category = self.nusc.get(
                    'category', instance['category_token'])['name']
                if category.startswith('vehicle.'):
                    low_vehicle_categories.append(category)
            result.update({
                'low_visibility_annotation_count': low_count,
                'low_visibility_vehicle_count': len(low_vehicle_categories),
                'low_visibility_vehicle_categories':
                    sorted(set(low_vehicle_categories)),
            })
        except Exception as error:
            result['annotation_error'] = repr(error)
        return result


def class_statistics(gt, baseline_pred, oup_pred, valid):
    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        gt_class = (gt == class_index) & valid
        baseline_class_error = (
            gt_class & (baseline_pred != class_index))
        oup_class_error = gt_class & (oup_pred != class_index)
        corrected = baseline_class_error & (oup_pred == class_index)
        new_errors = (
            gt_class & (baseline_pred == class_index)
            & (oup_pred != class_index))
        rows.append({
            'class_index': class_index,
            'class_name': class_name,
            'gt_voxels': int(gt_class.sum()),
            'baseline_errors_on_gt': int(baseline_class_error.sum()),
            'oup_errors_on_gt': int(oup_class_error.sum()),
            'corrected_voxels': int(corrected.sum()),
            'new_errors': int(new_errors.sum()),
            'net_corrected_voxels':
                int(corrected.sum()) - int(new_errors.sum()),
        })
    return rows


def metric_record(
        index, meta, gt, mask_camera, mask_lidar, baseline_pred, oup_pred,
        points, oup_output, annotation_evidence):
    valid_label = (
        (gt != IGNORE_INDEX) & (gt >= 0) & (gt < len(CLASS_NAMES)))
    valid = valid_label & mask_camera
    baseline_error = (baseline_pred != gt) & valid
    oup_error = (oup_pred != gt) & valid
    corrected = baseline_error & ~oup_error
    new_errors = ~baseline_error & oup_error & valid

    baseline_hist = confusion_matrix(baseline_pred, gt, valid)
    oup_hist = confusion_matrix(oup_pred, gt, valid)
    baseline_class_iou = per_class_iou(baseline_hist) * 100.0
    oup_class_iou = per_class_iou(oup_hist) * 100.0
    baseline_miou = float(np.nanmean(baseline_class_iou[:-1]))
    oup_miou = float(np.nanmean(oup_class_iou[:-1]))

    distances = distance_volume(gt.shape)
    distance_stats = {}
    for name, lower, upper in DISTANCE_BINS:
        region = valid & (distances >= lower) & (distances < upper)
        distance_stats[name] = {
            'valid_voxels': int(region.sum()),
            'gt_occupied_voxels': int((region & (gt != FREE_CLASS)).sum()),
            'baseline_error_rate': safe_divide(
                int((baseline_error & region).sum()), int(region.sum())),
            'oup_error_rate': safe_divide(
                int((oup_error & region).sum()), int(region.sum())),
            'corrected_voxels': int((corrected & region).sum()),
            'new_errors': int((new_errors & region).sum()),
        }

    boundary = road_boundary_mask(gt, valid)
    point_stats = point_density_statistics(points, gt, valid)
    sparse_volume = np.repeat(
        point_stats.pop('sparse_columns')[:, :, None],
        gt.shape[2],
        axis=2)
    vehicle = np.isin(gt, list(VEHICLE_CLASSES)) & valid
    small_object = np.isin(gt, list(SMALL_OBJECT_CLASSES)) & valid
    occlusion = get_occlusion_uncertainty(oup_output, gt.shape)
    mean_corrected_occlusion = float('nan')
    if occlusion is not None and corrected.any():
        mean_corrected_occlusion = float(occlusion[0][corrected].mean())

    class_rows = class_statistics(
        gt, baseline_pred, oup_pred, valid)
    improved_classes = sorted(
        class_rows[:-1],
        key=lambda row: (
            row['net_corrected_voxels'],
            row['corrected_voxels']),
        reverse=True)
    failed_classes = sorted(
        class_rows[:-1],
        key=lambda row: (
            row['oup_errors_on_gt'],
            row['new_errors']),
        reverse=True)

    sample_token = str(
        meta.get('sample_token')
        or meta.get('token')
        or meta.get('sample_idx')
        or 'sample_{:06d}'.format(index))
    scene_token = str(meta.get('scene_token') or '')
    annotation = annotation_evidence.describe(sample_token)
    visible_occupied = valid & (gt != FREE_CLASS)

    record = {
        'sample_index': int(index),
        'sample_token': sample_token,
        'scene_token': scene_token,
        'valid_voxels': int(valid.sum()),
        'baseline_occupancy_iou': occupancy_iou(
            baseline_pred, gt, valid),
        'oup_occupancy_iou': occupancy_iou(
            oup_pred, gt, valid),
        'baseline_miou': baseline_miou,
        'oup_miou': oup_miou,
        'occupancy_iou_gain': (
            occupancy_iou(oup_pred, gt, valid)
            - occupancy_iou(baseline_pred, gt, valid)),
        'miou_gain': oup_miou - baseline_miou,
        'baseline_error_rate': safe_divide(
            int(baseline_error.sum()), int(valid.sum())),
        'oup_error_rate': safe_divide(
            int(oup_error.sum()), int(valid.sum())),
        'error_rate_reduction': safe_divide(
            int(baseline_error.sum()) - int(oup_error.sum()),
            int(valid.sum())),
        'corrected_voxels': int(corrected.sum()),
        'new_errors': int(new_errors.sum()),
        'net_corrected_voxels':
            int(corrected.sum()) - int(new_errors.sum()),
        'main_improved_classes': [
            row['class_name'] for row in improved_classes
            if row['net_corrected_voxels'] > 0
        ][:5],
        'main_failure_classes': [
            row['class_name'] for row in failed_classes
            if row['oup_errors_on_gt'] > 0
        ][:5],
        'class_statistics': class_rows,
        'baseline_per_class_iou': {
            CLASS_NAMES[i]: finite_or_none(value)
            for i, value in enumerate(baseline_class_iou)
        },
        'oup_per_class_iou': {
            CLASS_NAMES[i]: finite_or_none(value)
            for i, value in enumerate(oup_class_iou)
        },
        'distance_statistics': distance_stats,
        'dominant_improvement_distance': max(
            distance_stats,
            key=lambda name: (
                distance_stats[name]['corrected_voxels']
                - distance_stats[name]['new_errors'])),
        'visibility_statistics': {
            'camera_visible_ratio': float(mask_camera.mean()),
            'lidar_visible_ratio': float(mask_lidar.mean()),
            'visible_by_both_ratio': float(
                (mask_camera & mask_lidar).mean()),
            'camera_only_ratio': float(
                (mask_camera & ~mask_lidar).mean()),
            'lidar_only_ratio': float(
                (~mask_camera & mask_lidar).mean()),
            'visible_occupied_voxels': int(visible_occupied.sum()),
        },
        'pointcloud_statistics': point_stats,
        'evidence_statistics': {
            'vehicle_corrected_voxels': int((vehicle & corrected).sum()),
            'vehicle_new_errors': int((vehicle & new_errors).sum()),
            'vehicle_sparse_corrected_voxels':
                int((vehicle & sparse_volume & corrected).sum()),
            'small_object_corrected_voxels':
                int((small_object & corrected).sum()),
            'small_object_new_errors':
                int((small_object & new_errors).sum()),
            'far_corrected_voxels':
                int((corrected & (distances >= 30.0)).sum()),
            'far_new_errors':
                int((new_errors & (distances >= 30.0)).sum()),
            'road_boundary_corrected_voxels':
                int((boundary & corrected).sum()),
            'road_boundary_new_errors':
                int((boundary & new_errors).sum()),
            'lidar_sparse_corrected_voxels':
                int((sparse_volume & corrected).sum()),
            'lidar_sparse_new_errors':
                int((sparse_volume & new_errors).sum()),
            'mean_occlusion_uncertainty_on_corrected':
                finite_or_none(mean_corrected_occlusion),
            **annotation,
        },
    }
    return record


def evidence_net(record, positive_key, negative_key):
    evidence = record['evidence_statistics']
    return evidence[positive_key] - evidence[negative_key]


def success_score(record, category, sparse_threshold):
    evidence = record['evidence_statistics']
    gain = max(0.0, record['miou_gain'])
    net = max(0, record['net_corrected_voxels'])
    if net <= 0:
        return None
    if category == 'occluded_vehicle':
        vehicle_net = evidence_net(
            record, 'vehicle_corrected_voxels', 'vehicle_new_errors')
        if (evidence['low_visibility_vehicle_count'] <= 0
                or vehicle_net <= 0):
            return None
        return (
            vehicle_net
            + 50.0 * evidence['low_visibility_vehicle_count']
            + 10.0 * gain)
    if category == 'far_range':
        category_net = evidence_net(
            record, 'far_corrected_voxels', 'far_new_errors')
        return None if category_net <= 0 else category_net + 10.0 * gain
    if category == 'small_object':
        category_net = evidence_net(
            record,
            'small_object_corrected_voxels',
            'small_object_new_errors')
        return None if category_net <= 0 else category_net + 10.0 * gain
    if category == 'road_boundary':
        category_net = evidence_net(
            record,
            'road_boundary_corrected_voxels',
            'road_boundary_new_errors')
        return None if category_net <= 0 else category_net + 10.0 * gain
    if category == 'lidar_sparse':
        density = record['pointcloud_statistics'][
            'points_per_visible_occupied_column']
        category_net = evidence_net(
            record,
            'lidar_sparse_corrected_voxels',
            'lidar_sparse_new_errors')
        if (not math.isfinite(density) or density > sparse_threshold
                or category_net <= 0):
            return None
        return (
            category_net + 100.0 * max(0.0, sparse_threshold - density)
            + 10.0 * gain)
    if category == 'overall_gain':
        return None if net <= 0 else net + 100.0 * gain
    raise KeyError(category)


def success_reason(record, category, sparse_threshold):
    evidence = record['evidence_statistics']
    common = (
        '实际预测：mIoU {:+.2f}，修正 {}、新增错误 {} 个体素'
        .format(
            record['miou_gain'],
            record['corrected_voxels'],
            record['new_errors']))
    if category == 'occluded_vehicle':
        return (
            '{}；nuScenes 标注含 {} 个低可见度车辆，车辆类修正 {} 个体素'
            .format(
                common,
                evidence['low_visibility_vehicle_count'],
                evidence['vehicle_corrected_voxels']))
    if category == 'far_range':
        return '{}；30m 外修正 {}、新增 {} 个体素'.format(
            common,
            evidence['far_corrected_voxels'],
            evidence['far_new_errors'])
    if category == 'small_object':
        return '{}；行人/自行车/摩托车/交通锥修正 {}、新增 {} 个体素'.format(
            common,
            evidence['small_object_corrected_voxels'],
            evidence['small_object_new_errors'])
    if category == 'road_boundary':
        return '{}；复杂道路边界修正 {}、新增 {} 个体素'.format(
            common,
            evidence['road_boundary_corrected_voxels'],
            evidence['road_boundary_new_errors'])
    if category == 'lidar_sparse':
        density = record['pointcloud_statistics'][
            'points_per_visible_occupied_column']
        return (
            '{}；LiDAR 点/可见占据柱 {:.2f}（稀疏阈值 {:.2f}），'
            '稀疏柱修正 {} 个体素'
        ).format(
            common,
            density,
            sparse_threshold,
            evidence['lidar_sparse_corrected_voxels'])
    return '{}；总体净修正 {} 个体素，主要改善类别为 {}'.format(
        common,
        record['net_corrected_voxels'],
        ', '.join(record['main_improved_classes']) or '无')


def failure_score(record, category, sparse_threshold):
    if category == 'regression':
        regression = record['new_errors'] - record['corrected_voxels']
        if regression <= 0 and record['miou_gain'] >= 0:
            return None
        return regression + 100.0 * max(0.0, -record['miou_gain'])
    if category == 'residual_error':
        return (
            10000.0 * record['oup_error_rate']
            + max(0, record['new_errors'] - record['corrected_voxels']))
    if category == 'insufficient_observation':
        density = record['pointcloud_statistics'][
            'points_per_visible_occupied_column']
        density_term = (
            max(0.0, sparse_threshold - density)
            if math.isfinite(density) else 0.0)
        visibility = record['visibility_statistics']['camera_visible_ratio']
        annotations = record['evidence_statistics'][
            'low_visibility_annotation_count']
        return (
            1000.0 * (1.0 - visibility)
            + 100.0 * density_term
            + 10.0 * annotations
            + 1000.0 * record['oup_error_rate'])
    raise KeyError(category)


def failure_reason(record, category, sparse_threshold):
    common = (
        '实际预测：OUP 错误率 {:.2%}，修正 {}、新增错误 {} 个体素'
        .format(
            record['oup_error_rate'],
            record['corrected_voxels'],
            record['new_errors']))
    if category == 'regression':
        return '{}；相对 Baseline mIoU {:+.2f}，净修正 {} 个体素'.format(
            common, record['miou_gain'], record['net_corrected_voxels'])
    if category == 'residual_error':
        return '{}；残余错误主要集中在 {}'.format(
            common,
            ', '.join(record['main_failure_classes']) or '无可识别类别')
    density = record['pointcloud_statistics'][
        'points_per_visible_occupied_column']
    visibility = record['visibility_statistics']['camera_visible_ratio']
    low_annotations = record['evidence_statistics'][
        'low_visibility_annotation_count']
    return (
        '{}；camera 可见比例 {:.2%}，LiDAR 点/可见占据柱 {:.2f}'
        '（稀疏阈值 {:.2f}），低可见度标注 {} 个'
    ).format(
        common,
        visibility,
        density,
        sparse_threshold,
        low_annotations)


def choose_one(
        records, score_fn, selected_tokens, selected_scenes,
        preferred_class_names=None):
    ranked = []
    for record in records:
        if record['sample_token'] in selected_tokens:
            continue
        score = score_fn(record)
        if score is None or not math.isfinite(score):
            continue
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return None

    preferred_class_names = preferred_class_names or set()
    for require_new_scene, require_new_class in [
            (True, True), (True, False), (False, True), (False, False)]:
        for _, record in ranked:
            if (require_new_scene and record['scene_token']
                    and record['scene_token'] in selected_scenes):
                continue
            if require_new_class:
                classes = set(record['main_improved_classes'])
                if classes and classes.issubset(preferred_class_names):
                    continue
            return record
    return ranked[0][1]


def select_cases(records, success_count, failure_count):
    densities = [
        record['pointcloud_statistics'][
            'points_per_visible_occupied_column']
        for record in records
        if math.isfinite(record['pointcloud_statistics'][
            'points_per_visible_occupied_column'])
    ]
    sparse_threshold = float(np.percentile(densities, 25)) if densities else 0.0

    selected_success = []
    selected_tokens = set()
    selected_scenes = set()
    selected_classes = set()
    success_categories = [
        'occluded_vehicle',
        'far_range',
        'small_object',
        'road_boundary',
        'lidar_sparse',
        'overall_gain',
    ]
    while len(success_categories) < success_count:
        success_categories.append('overall_gain')
    for category in success_categories[:success_count]:
        record = choose_one(
            records,
            lambda item, category=category: success_score(
                item, category, sparse_threshold),
            selected_tokens,
            selected_scenes,
            selected_classes)
        if record is None:
            record = choose_one(
                records,
                lambda item: (
                    item['net_corrected_voxels']
                    + 100.0 * max(0.0, item['miou_gain'])
                    if item['net_corrected_voxels'] > 0 else None),
                selected_tokens,
                selected_scenes,
                selected_classes)
            category = 'evidence_based_fallback'
        if record is None:
            raise RuntimeError('真实预测中不足以筛选 {} 个优势案例'.format(
                success_count))
        selected_tokens.add(record['sample_token'])
        selected_scenes.add(record['scene_token'])
        selected_classes.update(record['main_improved_classes'])
        selected = copy.deepcopy(record)
        selected['case_type'] = 'success'
        selected['selection_category'] = category
        if category == 'evidence_based_fallback':
            selected['selection_reason'] = (
                '指定类别证据不足时的实际正收益候选；mIoU {:+.2f}，'
                '净修正 {} 个体素'
            ).format(
                record['miou_gain'], record['net_corrected_voxels'])
        else:
            selected['selection_reason'] = success_reason(
                record, category, sparse_threshold)
        selected_success.append(selected)

    selected_failure = []
    failure_scenes = set(selected_scenes)
    failure_categories = [
        'regression', 'residual_error', 'insufficient_observation']
    while len(failure_categories) < failure_count:
        failure_categories.append('residual_error')
    for category in failure_categories[:failure_count]:
        record = choose_one(
            records,
            lambda item, category=category: failure_score(
                item, category, sparse_threshold),
            selected_tokens,
            failure_scenes,
            set())
        if record is None:
            raise RuntimeError('真实预测中不足以筛选 {} 个失败案例'.format(
                failure_count))
        selected_tokens.add(record['sample_token'])
        failure_scenes.add(record['scene_token'])
        selected = copy.deepcopy(record)
        selected['case_type'] = 'failure'
        selected['selection_category'] = category
        selected['selection_reason'] = failure_reason(
            record, category, sparse_threshold)
        selected_failure.append(selected)

    for index, record in enumerate(selected_success, 1):
        record['case_id'] = 'case_{:03d}'.format(index)
    for index, record in enumerate(selected_failure, 1):
        record['case_id'] = 'case_{:03d}'.format(index)
    return selected_success, selected_failure, sparse_threshold


def feature_map_to_xy(value, xy_shape, batch_index=0):
    if value is None:
        return None
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
    else:
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.dim() == 4:
        value = value[batch_index, 0]
    elif value.dim() == 3:
        value = value[batch_index]
    if value.dim() != 2:
        return None
    tensor = value.view(1, 1, value.shape[0], value.shape[1])
    resized = F.interpolate(
        tensor,
        size=(xy_shape[1], xy_shape[0]),
        mode='bilinear',
        align_corners=False)
    # Feature tensors are [H=y,W=x], labels are [X,Y].
    return resized[0, 0].numpy().T


def voxel_uncertainty_to_bev(value, spatial_shape):
    array = tensor_to_bxyz_numpy(value, spatial_shape)
    if array is None:
        return None
    return array[0].max(axis=2)


def scatter_query_values(indices, values, feature_hw, xy_shape):
    if indices is None or values is None:
        return None
    indices = np.asarray(indices).reshape(-1)
    values = np.asarray(values).reshape(-1)
    result = np.zeros(feature_hw, dtype=np.float32)
    valid = (indices >= 0) & (indices < feature_hw[0] * feature_hw[1])
    result.reshape(-1)[indices[valid]] = values[valid]
    return feature_map_to_xy(result, xy_shape)


def extract_intermediates(oup_output, gt_shape):
    arrays = {}
    availability = {
        key: False for key in REQUESTED_INTERMEDIATES
    }
    if not isinstance(oup_output, dict):
        return arrays, availability, {}

    uncertainty = oup_output.get('uncertainty_dict', {})
    reliability = oup_output.get('reliability_dict', {})
    udca = oup_output.get('udca_dict', {})
    xy_shape = gt_shape[:2]

    reliability_specs = [
        ('reliability_camera', 'R_cam'),
        ('reliability_lidar', 'R_lidar'),
    ]
    for output_name, key in reliability_specs:
        value = feature_map_to_xy(reliability.get(key), xy_shape)
        if value is not None:
            arrays[output_name] = value
            availability[output_name] = True
    discrepancy = reliability.get(
        'discrepancy_map', reliability.get('D_abs'))
    value = feature_map_to_xy(discrepancy, xy_shape)
    if value is not None:
        arrays['cross_modal_discrepancy'] = value
        availability['cross_modal_discrepancy'] = True

    uncertainty_specs = [
        ('uncertainty_semantic', 'U_sem_voxel'),
        ('uncertainty_confidence', 'U_conf_voxel'),
        ('uncertainty_occlusion', 'U_occ_voxel'),
    ]
    for output_name, key in uncertainty_specs:
        value = voxel_uncertainty_to_bev(
            uncertainty.get(key), gt_shape)
        if value is not None:
            arrays[output_name] = value
            availability[output_name] = True
    u_bev = uncertainty.get('U_total_bev')
    if u_bev is not None:
        u_bev_array = tensor_to_bxyz_numpy(
            uncertainty.get('U_total_voxel'), gt_shape)
        if u_bev_array is not None:
            arrays['uncertainty_bev'] = u_bev_array[0].max(axis=2)
            availability['uncertainty_bev'] = True

    selected_mask = udca.get('selected_query_mask')
    selected_indices = udca.get('selected_query_indices')
    if selected_mask is not None:
        mask_tensor = selected_mask.detach().float().cpu()
        feature_hw = tuple(mask_tensor.shape[-2:])
        arrays['query_locations'] = feature_map_to_xy(
            selected_mask, xy_shape)
        availability['query_locations'] = True
    else:
        feature_hw = None
    if selected_indices is not None:
        selected_indices_np = selected_indices[0].detach().cpu().numpy()
        arrays['selected_query_indices'] = selected_indices_np
    else:
        selected_indices_np = None

    for output_name, key in [
            ('camera_attention', 'attn_weights_cam'),
            ('lidar_attention', 'attn_weights_lidar')]:
        weights = udca.get(key)
        if (weights is not None and feature_hw is not None
                and selected_indices_np is not None):
            values = weights[0].detach().float().cpu().numpy().sum(axis=1)
            map_xy = scatter_query_values(
                selected_indices_np, values, feature_hw, xy_shape)
            arrays[output_name] = map_xy
            arrays[output_name + '_per_query'] = values
            availability[output_name] = True

    offsets = {}
    for source, key in [
            ('camera', 'offsets_cam'), ('lidar', 'offsets_lidar')]:
        value = udca.get(key)
        if value is not None:
            offsets[source] = value[0].detach().float().cpu().numpy()
            arrays['sampling_offsets_' + source] = offsets[source]
    if offsets and selected_indices_np is not None and feature_hw is not None:
        availability['sampling_offsets'] = True

    gate = feature_map_to_xy(
        uncertainty.get('gate_map'), xy_shape)
    if gate is not None:
        arrays['refinement_gate'] = gate
        availability['refinement_gate'] = True

    stage1 = coarse_predict_from_output(oup_output)
    if stage1 is not None:
        arrays['stage1_prediction'] = stage1[0]
        availability['stage1_prediction'] = True
    availability['final_prediction'] = True
    return arrays, availability, {
        'feature_hw': feature_hw,
        'selected_query_indices': selected_indices_np,
        'offsets': offsets,
    }


def save_figure(fig, path, dpi=240):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.03)
    plt.close(fig)


def save_heatmap(array, path, cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(
        array.T,
        origin='lower',
        extent=BEV_EXTENT,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest')
    ax.axis('off')
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.01)
    save_figure(fig, path)


def semantic_bev(label, valid):
    occupied = valid & (label != FREE_CLASS) & (label != IGNORE_INDEX)
    z_indices = np.arange(label.shape[2], dtype=np.int16)
    top_z = np.where(occupied, z_indices[None, None, :], -1).max(axis=2)
    bev = np.full(label.shape[:2], -1, dtype=np.int16)
    has_value = top_z >= 0
    if has_value.any():
        gathered = np.take_along_axis(
            label, np.maximum(top_z, 0)[..., None], axis=2)[..., 0]
        bev[has_value] = gathered[has_value]
    return bev


def save_semantic_bev(label, valid, path):
    bev = semantic_bev(label, valid)
    image = np.full((*bev.shape, 3), 255, dtype=np.uint8)
    for class_index in range(FREE_CLASS):
        image[bev == class_index] = CLASS_COLORS[class_index]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        image.transpose(1, 0, 2),
        origin='lower',
        extent=BEV_EXTENT,
        interpolation='nearest')
    ax.axis('off')
    save_figure(fig, path)


def save_binary_bev(mask, path, color):
    bev = mask.max(axis=2)
    image = np.full((*bev.shape, 4), 0.0, dtype=np.float32)
    image[..., :3] = 1.0
    image[..., 3] = 1.0
    image[bev, :3] = np.asarray(color, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        image.transpose(1, 0, 2),
        origin='lower',
        extent=BEV_EXTENT,
        interpolation='nearest')
    ax.axis('off')
    save_figure(fig, path)


def occupied_voxel_points(label, valid):
    indices = np.argwhere(
        valid & (label != FREE_CLASS) & (label != IGNORE_INDEX))
    if indices.size == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int16)
    shape = np.asarray(label.shape, dtype=np.float64)
    lower = np.asarray(POINT_CLOUD_RANGE[:3])
    upper = np.asarray(POINT_CLOUD_RANGE[3:])
    centers = lower + (indices + 0.5) * (upper - lower) / shape
    classes = label[indices[:, 0], indices[:, 1], indices[:, 2]]
    return centers, classes


def save_semantic_3d(label, valid, path, max_points):
    points, classes = occupied_voxel_points(label, valid)
    if points.shape[0] > max_points:
        indices = np.linspace(
            0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        classes = classes[indices]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    if points.size:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=CLASS_COLORS[classes] / 255.0,
            s=1.2,
            alpha=0.9,
            depthshade=False,
            linewidths=0)
    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_zlim(POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5])
    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect((80, 80, 18))
    ax.axis('off')
    save_figure(fig, path, dpi=260)


def save_pointcloud_bev(points, path):
    points = np.asarray(points)
    in_range = (
        (points[:, 0] >= POINT_CLOUD_RANGE[0])
        & (points[:, 0] <= POINT_CLOUD_RANGE[3])
        & (points[:, 1] >= POINT_CLOUD_RANGE[1])
        & (points[:, 1] <= POINT_CLOUD_RANGE[4]))
    points = points[in_range]
    colors = points[:, 3] if points.shape[1] > 3 else points[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=0.15,
        cmap='viridis',
        linewidths=0,
        rasterized=True)
    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_aspect('equal')
    ax.axis('off')
    save_figure(fig, path, dpi=260)


def save_pointcloud_3d(points, path, max_points):
    points = np.asarray(points)
    if points.shape[0] > max_points:
        indices = np.linspace(
            0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = points[:, 3] if points.shape[1] > 3 else points[:, 2]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.15,
        cmap='viridis',
        depthshade=False,
        linewidths=0)
    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_zlim(POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5])
    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect((80, 80, 18))
    ax.axis('off')
    save_figure(fig, path, dpi=260)


def save_sampling_offsets(arrays, metadata, path):
    indices = metadata.get('selected_query_indices')
    feature_hw = metadata.get('feature_hw')
    offsets = metadata.get('offsets', {})
    if indices is None or feature_hw is None or not offsets:
        return False
    h, w = feature_hw
    query_h = indices // w
    query_w = indices % w
    stride = max(1, len(indices) // 250)
    fig, ax = plt.subplots(figsize=(8, 8))
    background = arrays.get('uncertainty_bev')
    if background is not None:
        ax.imshow(
            background.T,
            origin='lower',
            extent=(0, w, 0, h),
            cmap='gray',
            alpha=0.7)
    for source, color in [('camera', '#00FFFF'), ('lidar', '#FFBF00')]:
        if source not in offsets:
            continue
        mean_offset = offsets[source].mean(axis=1)
        ax.quiver(
            query_w[::stride],
            query_h[::stride],
            mean_offset[::stride, 1],
            mean_offset[::stride, 0],
            color=color,
            angles='xy',
            scale_units='xy',
            scale=1.0,
            width=0.0025)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    save_figure(fig, path)
    return True


def save_class_legend(output_dir):
    handles = [
        Line2D(
            [0], [0],
            marker='s',
            linestyle='',
            color=CLASS_COLORS[index] / 255.0,
            label='{}: {}'.format(index, CLASS_NAMES[index]),
            markersize=10)
        for index in range(FREE_CLASS)
    ]
    fig = plt.figure(figsize=(12, 4))
    fig.legend(handles=handles, loc='center', ncol=6, frameon=False)
    save_figure(fig, os.path.join(output_dir, 'class_legend.png'), dpi=220)


def save_camera_inputs(data_info, case_dir, missing):
    cams = data_info.get('cams', {})
    for camera_name, output_name in CAMERA_FILES:
        camera = cams.get(camera_name)
        if camera is None:
            missing.append('{}: data_info 无 {}'.format(
                output_name, camera_name))
            continue
        source = camera.get('data_path')
        if not source:
            missing.append('{}: {} 无 data_path'.format(
                output_name, camera_name))
            continue
        source = absolute_path(source)
        image = mmcv.imread(source)
        if image is None:
            missing.append('{}: 无法读取 {}'.format(output_name, source))
            continue
        mmcv.imwrite(image, os.path.join(case_dir, output_name))


def save_intermediate_pngs(
        arrays, availability, metadata, gt, valid, case_dir):
    heatmap_specs = [
        ('reliability_camera', 'reliability_camera.png', 'viridis', 0.0, 1.0),
        ('reliability_lidar', 'reliability_lidar.png', 'viridis', 0.0, 1.0),
        (
            'cross_modal_discrepancy',
            'cross_modal_discrepancy.png',
            'magma',
            0.0,
            1.0),
        (
            'uncertainty_semantic',
            'uncertainty_semantic.png',
            'viridis',
            0.0,
            1.0),
        (
            'uncertainty_confidence',
            'uncertainty_confidence.png',
            'viridis',
            0.0,
            1.0),
        (
            'uncertainty_occlusion',
            'uncertainty_occlusion.png',
            'viridis',
            0.0,
            1.0),
        ('uncertainty_bev', 'uncertainty_bev.png', 'viridis', 0.0, 1.0),
        ('query_locations', 'query_locations.png', 'Reds', 0.0, 1.0),
        ('camera_attention', 'camera_attention.png', 'viridis', 0.0, None),
        ('lidar_attention', 'lidar_attention.png', 'viridis', 0.0, None),
        ('refinement_gate', 'refinement_gate.png', 'viridis', 0.0, 1.0),
    ]
    for key, filename, cmap, vmin, vmax in heatmap_specs:
        if availability.get(key) and arrays.get(key) is not None:
            save_heatmap(
                arrays[key],
                os.path.join(case_dir, filename),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)
    if availability.get('sampling_offsets'):
        saved = save_sampling_offsets(
            arrays,
            metadata,
            os.path.join(case_dir, 'sampling_offsets.png'))
        availability['sampling_offsets'] = saved
    if availability.get('stage1_prediction'):
        save_semantic_bev(
            arrays['stage1_prediction'],
            valid,
            os.path.join(case_dir, 'stage1_prediction.png'))
    if availability.get('final_prediction'):
        # The actual final prediction is added to arrays by export_case.
        save_semantic_bev(
            arrays['final_prediction'],
            valid,
            os.path.join(case_dir, 'final_prediction.png'))


def export_case(
        selected, dataset, baseline_model, oup_model, output_dir,
        max_3d_points):
    index = selected['sample_index']
    sample = dataset[index]
    data_batch = collate([sample], samples_per_gpu=1)
    metas = get_img_metas(data_batch)
    meta = metas[0] if metas else {}
    actual_token = str(
        meta.get('sample_idx')
        or meta.get('token')
        or 'sample_{:06d}'.format(index))
    if actual_token != selected['sample_token']:
        raise RuntimeError(
            '重跑所选样本时 token 不一致: {} vs {}'.format(
                actual_token, selected['sample_token']))

    gt = require_batch_tensor(data_batch, 'voxel_semantics')[0].numpy()
    mask_camera = require_batch_tensor(data_batch, 'mask_camera')[0].bool().numpy()
    mask_lidar = require_batch_tensor(data_batch, 'mask_lidar')[0].bool().numpy()
    points = get_points(data_batch)
    baseline_pred, oup_pred, oup_output = infer_pair(
        baseline_model, oup_model, data_batch)
    baseline_pred = baseline_pred[0]
    oup_pred = oup_pred[0]
    valid = (
        mask_camera
        & (gt != IGNORE_INDEX)
        & (gt >= 0)
        & (gt < len(CLASS_NAMES)))
    improved = (baseline_pred != gt) & (oup_pred == gt) & valid
    degraded = (baseline_pred == gt) & (oup_pred != gt) & valid

    arrays, availability, intermediate_meta = extract_intermediates(
        oup_output, gt.shape)
    arrays.update({
        'baseline_prediction': baseline_pred,
        'oup_prediction': oup_pred,
        'ground_truth': gt,
        'mask_camera': mask_camera.astype(np.uint8),
        'mask_lidar': mask_lidar.astype(np.uint8),
        'improved_voxels': improved.astype(np.uint8),
        'degraded_voxels': degraded.astype(np.uint8),
        'pointcloud': points,
        'final_prediction': oup_pred,
    })
    del oup_output

    case_dir = os.path.join(
        output_dir,
        'success_cases' if selected['case_type'] == 'success'
        else 'failure_cases',
        selected['case_id'])
    mmcv.mkdir_or_exist(case_dir)
    missing = []
    data_info = dataset.data_infos[index]
    save_camera_inputs(data_info, case_dir, missing)
    save_pointcloud_bev(
        points, os.path.join(case_dir, 'pointcloud_bev.png'))
    save_pointcloud_3d(
        points,
        os.path.join(case_dir, 'pointcloud_3d.png'),
        max_3d_points)

    for label, filename in [
            (baseline_pred, 'baseline_prediction.png'),
            (oup_pred, 'oup_prediction.png'),
            (gt, 'ground_truth.png')]:
        save_semantic_3d(
            label,
            valid,
            os.path.join(case_dir, filename),
            max_3d_points)
    for label, filename in [
            (baseline_pred, 'baseline_bev.png'),
            (oup_pred, 'oup_bev.png'),
            (gt, 'ground_truth_bev.png')]:
        save_semantic_bev(label, valid, os.path.join(case_dir, filename))
    save_binary_bev(
        improved,
        os.path.join(case_dir, 'improved_voxels.png'),
        [0.0, 0.75, 0.15])
    save_binary_bev(
        degraded,
        os.path.join(case_dir, 'degraded_voxels.png'),
        [0.9, 0.05, 0.05])
    save_intermediate_pngs(
        arrays,
        availability,
        intermediate_meta,
        gt,
        valid,
        case_dir)
    np.savez_compressed(
        os.path.join(case_dir, 'arrays.npz'),
        **arrays)
    with open(
            os.path.join(case_dir, 'case_metadata.json'),
            'w',
            encoding='utf-8') as file:
        json.dump(
            json_safe({
                'selection': selected,
                'intermediate_availability': availability,
                'missing_files': missing,
            }),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False)
    return availability, missing


def flatten_selection_row(record):
    visibility = record['visibility_statistics']
    pointcloud = record['pointcloud_statistics']
    evidence = record['evidence_statistics']
    return {
        'case_type': record['case_type'],
        'case_id': record['case_id'],
        'selection_category': record['selection_category'],
        'sample_index': record['sample_index'],
        'sample_token': record['sample_token'],
        'scene_token': record['scene_token'],
        'baseline_occupancy_iou': record['baseline_occupancy_iou'],
        'oup_occupancy_iou': record['oup_occupancy_iou'],
        'occupancy_iou_gain': record['occupancy_iou_gain'],
        'baseline_miou': record['baseline_miou'],
        'oup_miou': record['oup_miou'],
        'miou_gain': record['miou_gain'],
        'baseline_error_rate': record['baseline_error_rate'],
        'oup_error_rate': record['oup_error_rate'],
        'error_rate_reduction': record['error_rate_reduction'],
        'corrected_voxels': record['corrected_voxels'],
        'new_errors': record['new_errors'],
        'net_corrected_voxels': record['net_corrected_voxels'],
        'main_improved_classes': ';'.join(
            record['main_improved_classes']),
        'main_failure_classes': ';'.join(
            record['main_failure_classes']),
        'dominant_improvement_distance':
            record['dominant_improvement_distance'],
        'camera_visible_ratio': visibility['camera_visible_ratio'],
        'lidar_visible_ratio': visibility['lidar_visible_ratio'],
        'camera_only_ratio': visibility['camera_only_ratio'],
        'points_per_visible_occupied_column':
            pointcloud['points_per_visible_occupied_column'],
        'sparse_column_ratio_on_visible_occupancy':
            pointcloud['sparse_column_ratio_on_visible_occupancy'],
        'low_visibility_vehicle_count':
            evidence['low_visibility_vehicle_count'],
        'selection_reason': record['selection_reason'],
    }


def write_selection_csv(path, selections):
    rows = [flatten_selection_row(record) for record in selections]
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_availability(case_exports):
    result = {}
    for key in REQUESTED_INTERMEDIATES:
        available_count = sum(
            int(item['availability'].get(key, False))
            for item in case_exports)
        result[key] = {
            'available_for_all_cases':
                available_count == len(case_exports),
            'available_case_count': available_count,
            'total_case_count': len(case_exports),
        }
    return result


def write_readme(
        path, paths, selections, sparse_threshold, availability,
        export_warnings, processed_count, val_size, elapsed_seconds,
        annotation_error):
    lines = [
        '# Occ3D-nuScenes 定性案例筛选',
        '',
        '本目录由 `tools/export_qualitative_cases.py` 从实际模型预测自动生成；'
        '未重新训练，也未修改配置或 checkpoint。',
        '',
        '## 实际使用文件',
        '',
        '- Baseline config: `{}`'.format(paths['baseline_config']),
        '- Baseline checkpoint: `{}`'.format(paths['baseline_checkpoint']),
        '- OUP-Occ config: `{}`'.format(paths['oup_config']),
        '- OUP-Occ checkpoint: `{}`'.format(paths['oup_checkpoint']),
        '',
        '## 评价与筛选口径',
        '',
        '- 数据：Occ3D-nuScenes val，处理 {}/{} 个样本。'.format(
            processed_count, val_size),
        '- 所有逐样本指标仅使用官方 `mask_camera` 有效体素。',
        '- `occupancy IoU` 为 occupied/free 二值 IoU；`mIoU` 为语义类'
        ' 0–16 的逐样本平均 IoU，排除 free 类 17。',
        '- 修正体素：Baseline 错且 OUP-Occ 对；新增错误：Baseline 对且'
        ' OUP-Occ 错。',
        '- 优势案例采用类别配额（低可见度车辆、远距离、小目标、道路边界、'
        'LiDAR 稀疏、总体收益）并优先避免重复 scene 与重复主改善类别，'
        '不是单一分数 Top-K。',
        '- LiDAR 稀疏阈值来自全体实际样本 `点数/可见占据柱` 的第 25 百分位：'
        ' {:.4f}。'.format(sparse_threshold),
        '- 遮挡车辆证据来自 nuScenes `visibility_token` 1/2 的车辆标注；'
        '仅在标注证据可用且该样本车辆体素确有修正时使用该描述。',
        '- 运行耗时：{:.1f} 分钟。'.format(elapsed_seconds / 60.0),
        '',
        '## 入选案例',
        '',
        '| 类型 | 案例 | sample token | scene token | '
        'Baseline mIoU | OUP mIoU | ΔmIoU | 修正/新增 | 入选依据 |',
        '|---|---|---|---|---:|---:|---:|---:|---|',
    ]
    for record in selections:
        lines.append(
            '| {} | {} | `{}` | `{}` | {:.2f} | {:.2f} | {:+.2f} | '
            '{}/{} | {} |'.format(
                record['case_type'],
                record['case_id'],
                record['sample_token'],
                record['scene_token'],
                record['baseline_miou'],
                record['oup_miou'],
                record['miou_gain'],
                record['corrected_voxels'],
                record['new_errors'],
                record['selection_reason'].replace('|', '/')))

    lines.extend([
        '',
        '## 图像与原始数组',
        '',
        '- 六相机 PNG 为该 nuScenes 样本的原始相机帧；模型推理仍使用配置中'
        '的确定性 val resize/crop。',
        '- `baseline_prediction.png`、`oup_prediction.png` 与'
        ' `ground_truth.png` 使用相同体素范围、类别颜色、视角和'
        ' `mask_camera` 区域。',
        '- 所有图均为独立 PNG，没有自动拼接总图。',
        '- 每个案例的 `arrays.npz` 保存完整预测、GT、mask、点云、'
        '改善/退化体素和实际可提取的中间数组。',
        '- 根目录 `class_legend.png` 是所有语义图共用的颜色图例。',
        '',
        '## 中间结果可用性',
        '',
    ])
    for key, value in availability.items():
        status = (
            '全部可用'
            if value['available_for_all_cases']
            else '仅 {}/{} 个案例可用'.format(
                value['available_case_count'],
                value['total_case_count']))
        lines.append('- `{}`：{}'.format(key, status))
    lines.extend([
        '',
        '实现说明：当前仓库没有名为 `RAPR` 的独立模块；这里把'
        ' `coarse_occ_logits` 作为粗预测阶段 `stage1_prediction`，'
        '把最终 logits 作为 `final_prediction`。'
        ' `UncertaintyGuidedBEVRefinement` 只返回最后一次 refinement'
        ' 的 `gate_map`，不会伪造每个 residual stage 后不存在的独立预测'
        '或 gate。CRF/UCRF、OUE 和 UDCA 字段均直接来自模型'
        ' `return_result=True` 输出，没有用猜测值替代。',
    ])
    if annotation_error:
        lines.extend([
            '',
            '### 标注证据限制',
            '',
            '- nuScenes visibility 标注初始化失败：`{}`。因此输出不会'
            '把任何样本描述为“已由标注确认的遮挡车辆”。'.format(
                annotation_error),
        ])
    if export_warnings:
        lines.extend(['', '## 导出警告', ''])
        for warning in export_warnings:
            lines.append('- {}'.format(warning))
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if args.success_count <= 0 or args.failure_count <= 0:
        raise ValueError('success-count 和 failure-count 必须大于 0')
    if not torch.cuda.is_available():
        raise RuntimeError('完整模型推理要求 CUDA GPU')

    paths = resolve_all_paths(args)
    output_dir = absolute_path(args.output_dir)
    mmcv.mkdir_or_exist(output_dir)
    mmcv.mkdir_or_exist(os.path.join(output_dir, 'success_cases'))
    mmcv.mkdir_or_exist(os.path.join(output_dir, 'failure_cases'))

    torch.cuda.set_device(args.gpu_id)
    set_random_seed(args.seed, deterministic=True)
    baseline_cfg = load_cfg(paths['baseline_config'])
    oup_cfg = load_cfg(paths['oup_config'])

    baseline_val = baseline_cfg.data.val
    oup_val = oup_cfg.data.val
    for key in ['type', 'ann_file', 'data_root']:
        if str(baseline_val.get(key)) != str(oup_val.get(key)):
            raise ValueError(
                'Baseline/OUP val {} 不一致: {} vs {}'.format(
                    key, baseline_val.get(key), oup_val.get(key)))

    dataset_cfg = patch_analysis_pipeline(oup_cfg.data.val)
    dataset = build_dataset(dataset_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)
    annotation_evidence = NuScenesAnnotationEvidence(
        dataset_cfg.get('data_root', 'data/nuscenes'))
    baseline_model = build_eval_model(
        baseline_cfg,
        paths['baseline_checkpoint'],
        dataset,
        args.gpu_id,
        'Baseline')
    oup_model = build_eval_model(
        oup_cfg,
        paths['oup_checkpoint'],
        dataset,
        args.gpu_id,
        'OUP-Occ')

    start_time = time.time()
    records = []
    sample_limit = (
        len(dataset)
        if args.max_samples is None or args.max_samples < 0
        else min(len(dataset), args.max_samples))
    print('开始遍历 Occ3D val: {} / {} 个样本'.format(
        sample_limit, len(dataset)))
    for index, data_batch in enumerate(loader):
        if index >= sample_limit:
            break
        metas = get_img_metas(data_batch)
        meta = metas[0] if metas else {}
        gt = require_batch_tensor(
            data_batch, 'voxel_semantics')[0].numpy()
        mask_camera = require_batch_tensor(
            data_batch, 'mask_camera')[0].bool().numpy()
        mask_lidar = require_batch_tensor(
            data_batch, 'mask_lidar')[0].bool().numpy()
        points = get_points(data_batch)
        baseline_pred, oup_pred, oup_output = infer_pair(
            baseline_model, oup_model, data_batch)
        records.append(metric_record(
            index=index,
            meta=meta,
            gt=gt,
            mask_camera=mask_camera,
            mask_lidar=mask_lidar,
            baseline_pred=baseline_pred[0],
            oup_pred=oup_pred[0],
            points=points,
            oup_output=oup_output,
            annotation_evidence=annotation_evidence))
        del oup_output
        if ((index + 1) % args.log_interval == 0
                or index + 1 == sample_limit):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                '已处理 {}/{}，平均 {:.3f}s/样本，预计剩余 {:.1f} 分钟'
                .format(
                    index + 1,
                    sample_limit,
                    elapsed / (index + 1),
                    (sample_limit - index - 1)
                    * elapsed / (index + 1) / 60.0))

    if len(records) < args.success_count + args.failure_count:
        raise RuntimeError('处理样本不足，无法生成要求数量的案例')
    success, failure, sparse_threshold = select_cases(
        records, args.success_count, args.failure_count)
    selections = success + failure

    save_class_legend(output_dir)
    case_exports = []
    export_warnings = []
    print('开始重跑并导出 {} 个所选案例'.format(len(selections)))
    for record in selections:
        availability, missing = export_case(
            record,
            dataset,
            baseline_model,
            oup_model,
            output_dir,
            args.max_3d_points)
        case_exports.append({
            'sample_token': record['sample_token'],
            'availability': availability,
        })
        for item in missing:
            export_warnings.append(
                '{} {}: {}'.format(
                    record['case_type'], record['case_id'], item))
        print('已导出 {} {}: {}'.format(
            record['case_type'],
            record['case_id'],
            record['sample_token']))

    availability = aggregate_availability(case_exports)
    elapsed_seconds = time.time() - start_time
    summary = {
        'methodology': {
            'dataset': 'Occ3D-nuScenes val',
            'processed_samples': len(records),
            'full_val_size': len(dataset),
            'official_evaluation_mask': 'mask_camera',
            'miou_excludes_free_class': True,
            'free_class_index': FREE_CLASS,
            'selection_is_prediction_based': True,
            'success_categories': [
                'occluded_vehicle', 'far_range', 'small_object',
                'road_boundary', 'lidar_sparse', 'overall_gain'
            ],
            'scene_diversity_preferred': True,
            'lidar_sparse_threshold_points_per_visible_occupied_column':
                sparse_threshold,
        },
        'resolved_paths': paths,
        'intermediate_availability': availability,
        'export_warnings': export_warnings,
        'success_cases': success,
        'failure_cases': failure,
    }
    with open(
            os.path.join(output_dir, 'selection_summary.json'),
            'w',
            encoding='utf-8') as file:
        json.dump(
            json_safe(summary),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False)
    write_selection_csv(
        os.path.join(output_dir, 'selection_summary.csv'),
        selections)
    write_readme(
        os.path.join(output_dir, 'README.md'),
        paths,
        selections,
        sparse_threshold,
        availability,
        export_warnings,
        len(records),
        len(dataset),
        elapsed_seconds,
        annotation_evidence.error)
    print('完成。结果目录: {}'.format(output_dir))
    print('一键完整运行命令: python tools/export_qualitative_cases.py --gpu-id {}'
          .format(args.gpu_id))


if __name__ == '__main__':
    main()
