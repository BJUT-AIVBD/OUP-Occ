#!/usr/bin/env python3
"""Evaluate trained Baseline/OUP-Occ models under deterministic corruption."""

import argparse
import json
import math
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mmcv  # noqa: E402
import numpy as np  # noqa: E402

from analysis_4_4_common import (  # noqa: E402
    CONDITION_SPECS,
    DEFAULT_BASELINE_CHECKPOINT,
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_OUP_CHECKPOINT,
    DEFAULT_OUP_CONFIG,
    EXPECTED_FULL_VAL_SIZE,
    RuntimeLogger,
    absolute_path,
    batch_sensor_hashes,
    build_analysis_dataset,
    build_analysis_loader,
    build_eval_model,
    configure_runtime,
    confusion_matrix,
    degradation_audit_from_batch,
    extract_gt_masks,
    finish_message,
    fmt,
    infer_model,
    install_sensor_corruption,
    load_cfg,
    markdown_table,
    metrics_from_hist,
    official_protocol,
    output_structure,
    predict_from_output,
    print_protocol,
    resolve_pair_paths,
    sample_limit,
    sample_token,
    validate_same_dataset,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = 'work_dirs/analysis_4_4/4_4_3_sensor_degradation'
DEFAULT_CONDITIONS = list(CONDITION_SPECS.keys())
CONDITION_NAMES = {
    'clean': '无退化',
    'lidar_drop25': 'LiDAR 随机丢弃 25%',
    'lidar_drop50': 'LiDAR 随机丢弃 50%',
    'camera_drop1': '随机关闭 1 个相机',
    'camera_drop3': '随机关闭 3 个相机',
    'brightness_0.5': '图像亮度 ×0.5',
    'gaussian_blur': '高斯模糊 k=7, σ=1.5',
    'combined': '关闭 1 相机 + LiDAR 丢弃 50%',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            '4.4.3 使用现有 checkpoint，在原始像素 Normalize 前和点云 '
            'voxelization 前施加确定性退化，比较相对同次 clean 的性能下降'))
    parser.add_argument(
        '--baseline-config', default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument(
        '--baseline-checkpoint', default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument('--oup-config', default=DEFAULT_OUP_CONFIG)
    parser.add_argument('--oup-checkpoint', default=DEFAULT_OUP_CHECKPOINT)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument(
        '--official-mask',
        default='camera',
        choices=['camera', 'lidar', 'both', 'union', 'all'])
    parser.add_argument(
        '--conditions',
        nargs='+',
        choices=DEFAULT_CONDITIONS,
        default=DEFAULT_CONDITIONS)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--full-val', action='store_true')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        help='smoke test 样本数；--full-val 时忽略')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='跳过签名一致且已经完整运行的 condition')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    return parser.parse_args()


def ordered_conditions(requested):
    unique = []
    for condition in requested:
        if condition not in unique:
            unique.append(condition)
    if 'clean' in unique:
        unique.remove('clean')
    return ['clean'] + unique


def run_signature(
        condition, args, paths, dataset_size, expected_samples, protocol):
    return {
        'implementation_version': 1,
        'condition': condition,
        'seed': args.seed,
        'baseline_config': paths['baseline_config'],
        'baseline_checkpoint': paths['baseline_checkpoint'],
        'oup_config': paths['oup_config'],
        'oup_checkpoint': paths['oup_checkpoint'],
        'official_mask': args.official_mask,
        'dataset_size': dataset_size,
        'expected_samples': expected_samples,
        'num_classes': protocol['num_classes'],
        'ignore_index': protocol['ignore_index'],
    }


def load_resumable_condition(path, expected_signature):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        value = json.load(file)
    if value.get('run_signature') != expected_signature:
        return None
    if value.get('processed_samples') != expected_signature['expected_samples']:
        return None
    if not value.get('complete', False):
        return None
    return value


def aggregate_audit(state, audit):
    original_points = int(audit.get('original_points', 0))
    remaining_points = int(audit.get('remaining_points', 0))
    state['original_points'] += original_points
    state['remaining_points'] += remaining_points
    state['dropped_points'] += max(0, original_points - remaining_points)
    failed = list(audit.get('failed_cameras', []))
    state['failed_camera_events'] += len(failed)
    for name in failed:
        state['failed_camera_counts'][name] = (
            state['failed_camera_counts'].get(name, 0) + 1)
    for record in audit.get('image_records', []):
        state['image_count'] += 1
        state['original_image_mean_sum'] += float(
            record.get('original_mean', 0.0))
        state['corrupted_image_mean_sum'] += float(
            record.get('corrupted_mean', 0.0))


def finalize_audit(state, first_audit, first_hashes):
    image_count = state['image_count']
    original_mean = (
        state['original_image_mean_sum'] / image_count
        if image_count else float('nan'))
    corrupted_mean = (
        state['corrupted_image_mean_sum'] / image_count
        if image_count else float('nan'))
    return {
        'total_original_points': state['original_points'],
        'total_remaining_points': state['remaining_points'],
        'total_dropped_points': state['dropped_points'],
        'actual_point_drop_rate': (
            state['dropped_points'] / state['original_points']
            if state['original_points'] else float('nan')),
        'failed_camera_events': state['failed_camera_events'],
        'failed_camera_counts': state['failed_camera_counts'],
        'mean_original_image_pixel': original_mean,
        'mean_corrupted_image_pixel': corrupted_mean,
        'mean_image_change': corrupted_mean - original_mean,
        'first_sample': first_audit,
        'first_sample_sensor_hashes': first_hashes,
        'paired_input_guarantee': (
            '每个样本仅构造一个 canonical data_batch，并依次传给两模型；'
            '首样本前后 hash 核验模型未原地修改传感器输入。'),
    }


def print_first_audit(logger, condition, audit, hashes):
    image_records = audit.get('image_records', [])
    before_mean = (
        float(np.mean([
            record['original_mean'] for record in image_records]))
        if image_records else float('nan'))
    after_mean = (
        float(np.mean([
            record['corrupted_mean'] for record in image_records]))
        if image_records else float('nan'))
    logger.log(
        '{} 首样本核验: token={}，点数 {}/{}，失效相机={}，'
        '图像均值 {:.4f}->{:.4f}，points/img/depth hash={}'.format(
            condition,
            audit.get('sample_token'),
            audit.get('original_points'),
            audit.get('remaining_points'),
            audit.get('failed_cameras', []),
            before_mean,
            after_mean,
            {
                'points': hashes.get('points_sha256', '')[:12],
                'images': hashes.get('images_sha256', '')[:12],
                'depth': hashes.get('gt_depth_sha256', '')[:12],
            }))


def evaluate_condition(
        condition,
        args,
        oup_cfg,
        dataset,
        loader,
        protocol,
        baseline_model,
        oup_model,
        paths,
        limit,
        logger):
    num_classes = protocol['num_classes']
    baseline_hist = np.zeros(
        (num_classes, num_classes), dtype=np.int64)
    oup_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_voxels = 0
    audit_state = {
        'original_points': 0,
        'remaining_points': 0,
        'dropped_points': 0,
        'failed_camera_events': 0,
        'failed_camera_counts': {},
        'image_count': 0,
        'original_image_mean_sum': 0.0,
        'corrupted_image_mean_sum': 0.0,
    }
    first_audit = None
    first_hashes = None
    forward_structures = None
    start = time.time()
    logger.log(
        '开始 condition={}，配置={}，样本={}/{}'.format(
            condition, CONDITION_SPECS[condition], limit, len(dataset)))

    processed = 0
    for index, data_batch in enumerate(loader):
        if index >= limit:
            break
        audit = degradation_audit_from_batch(data_batch)
        hashes_before = None
        if index == 0:
            hashes_before = batch_sensor_hashes(data_batch)
            first_audit = audit
            first_hashes = dict(hashes_before)
            print_first_audit(logger, condition, audit, hashes_before)
        aggregate_audit(audit_state, audit)

        baseline_output = infer_model(baseline_model, data_batch)
        baseline_pred = predict_from_output(
            baseline_output, num_classes, prefer='final')[0]
        baseline_structure = (
            output_structure(baseline_output) if index == 0 else None)
        del baseline_output
        if index == 0:
            hashes_after_baseline = batch_sensor_hashes(data_batch)
            if hashes_after_baseline != hashes_before:
                raise RuntimeError(
                    'Baseline 前向原地修改了 canonical 传感器输入')

        oup_output = infer_model(oup_model, data_batch)
        oup_pred = predict_from_output(
            oup_output, num_classes, prefer='final')[0]
        if index == 0:
            forward_structures = {
                'baseline': baseline_structure,
                'oup': output_structure(oup_output),
            }
        del oup_output
        if index == 0:
            hashes_after_oup = batch_sensor_hashes(data_batch)
            if hashes_after_oup != hashes_before:
                raise RuntimeError(
                    'OUP-Occ 前向原地修改了 canonical 传感器输入')
            first_hashes['unchanged_after_both_models'] = True

        if baseline_pred.shape != oup_pred.shape:
            raise ValueError('两模型预测 shape 不一致')
        spatial_shape = tuple(oup_pred.shape)
        gt, _, _, valid, _ = extract_gt_masks(
            data_batch,
            spatial_shape,
            num_classes,
            protocol['ignore_index'],
            args.official_mask)
        baseline_hist += confusion_matrix(
            baseline_pred, gt, valid, num_classes)
        oup_hist += confusion_matrix(
            oup_pred, gt, valid, num_classes)
        valid_voxels += int(valid.sum())
        processed += 1

        if processed % args.log_interval == 0 or processed == limit:
            elapsed = max(time.time() - start, 1e-6)
            logger.log(
                '{} 已处理 {}/{}，平均 {:.3f}s/样本，最近 token={}'.format(
                    condition,
                    processed,
                    limit,
                    elapsed / processed,
                    sample_token(data_batch, index)))

    if processed != limit:
        raise RuntimeError(
            '{} 仅处理 {}/{} 个样本，不允许静默跳过'.format(
                condition, processed, limit))
    baseline_metrics = metrics_from_hist(
        baseline_hist, protocol['free_class_index'])
    oup_metrics = metrics_from_hist(
        oup_hist, protocol['free_class_index'])
    per_class = []
    for class_index, class_name in enumerate(protocol['class_names']):
        per_class.append({
            'condition': condition,
            'class_index': class_index,
            'class_name': class_name,
            'included_in_miou': (
                class_index < protocol['free_class_index']),
            'baseline_iou': baseline_metrics[
                'per_class_iou'][class_index],
            'oup_iou': oup_metrics['per_class_iou'][class_index],
            'oup_advantage': (
                oup_metrics['per_class_iou'][class_index]
                - baseline_metrics['per_class_iou'][class_index]),
        })
    return {
        'condition': condition,
        'condition_cn': CONDITION_NAMES[condition],
        'corruption_config': CONDITION_SPECS[condition],
        'run_signature': run_signature(
            condition,
            args,
            paths,
            len(dataset),
            limit,
            protocol),
        'processed_samples': processed,
        'valid_voxels': valid_voxels,
        'baseline_miou': baseline_metrics['miou'],
        'oup_miou': oup_metrics['miou'],
        'oup_advantage_over_baseline': (
            oup_metrics['miou'] - baseline_metrics['miou']),
        'per_class': per_class,
        'degradation_audit': finalize_audit(
            audit_state, first_audit, first_hashes),
        'forward_output_structure': forward_structures,
        'runtime_seconds': time.time() - start,
        'complete': True,
    }


def summary_rows(completed):
    if 'clean' not in completed:
        raise RuntimeError('必须先得到同次 clean 结果')
    clean = completed['clean']
    clean_baseline = clean['baseline_miou']
    clean_oup = clean['oup_miou']
    rows = []
    for condition, value in completed.items():
        rows.append({
            'condition': condition,
            'condition_cn': CONDITION_NAMES[condition],
            'processed_samples': value['processed_samples'],
            'valid_voxels': value['valid_voxels'],
            'baseline_miou': value['baseline_miou'],
            'oup_miou': value['oup_miou'],
            'baseline_absolute_drop': (
                clean_baseline - value['baseline_miou']),
            'oup_absolute_drop': clean_oup - value['oup_miou'],
            'baseline_retention_percent': (
                value['baseline_miou'] / clean_baseline * 100.0
                if clean_baseline > 0 else float('nan')),
            'oup_retention_percent': (
                value['oup_miou'] / clean_oup * 100.0
                if clean_oup > 0 else float('nan')),
            'oup_advantage_over_baseline': (
                value['oup_miou'] - value['baseline_miou']),
            'actual_point_drop_rate': value[
                'degradation_audit']['actual_point_drop_rate'],
            'failed_camera_events': value[
                'degradation_audit']['failed_camera_events'],
            'mean_original_image_pixel': value[
                'degradation_audit']['mean_original_image_pixel'],
            'mean_corrupted_image_pixel': value[
                'degradation_audit']['mean_corrupted_image_pixel'],
            'runtime_seconds': value['runtime_seconds'],
        })
    return rows


def write_current_outputs(out_dir, paths, protocol, checkpoint_audit, completed):
    rows = summary_rows(completed)
    per_class = []
    for value in completed.values():
        per_class.extend(value['per_class'])
    result = {
        'resolved_paths': paths,
        'evaluation_protocol': protocol,
        'checkpoint_audit': checkpoint_audit,
        'conditions': completed,
        'summary': rows,
        'resume_files': [
            'condition_{}.json'.format(name) for name in completed],
    }
    write_json(
        os.path.join(out_dir, 'sensor_degradation_robustness.json'),
        result)
    write_csv(
        os.path.join(out_dir, 'sensor_degradation_summary.csv'), rows)
    write_csv(
        os.path.join(out_dir, 'sensor_degradation_per_class.csv'),
        per_class)
    return result, rows


def make_plots(out_dir, rows):
    labels = [row['condition'] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - width / 2,
        [row['baseline_miou'] for row in rows],
        width,
        label='Baseline')
    ax.bar(
        x + width / 2,
        [row['oup_miou'] for row in rows],
        width,
        label='OUP-Occ')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('mIoU under Sensor Degradation')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'degradation_miou.png'), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        [row['baseline_retention_percent'] for row in rows],
        marker='o',
        label='Baseline')
    ax.plot(
        x,
        [row['oup_retention_percent'] for row in rows],
        marker='o',
        label='OUP-Occ')
    ax.axhline(100.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Retention (%)')
    ax.set_title('Performance Retention Relative to Clean')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'degradation_retention.png'), dpi=180)
    plt.close(fig)


def category_analysis(rows, conditions, category):
    subset = [row for row in rows if row['condition'] in conditions]
    if not subset:
        return '- {}：本次命令未评测对应条件。'.format(category)
    details = []
    for row in subset:
        baseline_drop = row['baseline_absolute_drop']
        oup_drop = row['oup_absolute_drop']
        if oup_drop < baseline_drop:
            comparison = 'OUP 下降更小'
        elif oup_drop > baseline_drop:
            comparison = 'OUP 下降更大'
        else:
            comparison = '两者下降相同'
        details.append(
            '`{}`：Baseline/OUP 下降 `{}`/`{}`，{}'.format(
                row['condition'],
                fmt(baseline_drop, 4),
                fmt(oup_drop, 4),
                comparison))
    return '- {}：{}。'.format(category, '；'.join(details))


def write_markdown(path, result, rows):
    protocol = result['evaluation_protocol']
    lines = [
        '# 4.4.3 传感器退化条件下的鲁棒性分析',
        '',
        '所有结果均由现有 checkpoint 在测试阶段直接推理得到；未微调、'
        '未训练，也未修改磁盘原始数据。',
        '',
        '## 评价与退化协议',
        '',
        '- Occ3D-nuScenes validation：实际每个 condition 处理 `{}` / `{}`'
        ' 个样本，评价 mask 为 `{}`，类别数 `{}`，ignore index `{}`。'
        .format(
            protocol['processed_samples_per_condition'],
            protocol['dataset_samples'],
            protocol['mask_tensor_key'],
            protocol['num_classes'],
            protocol['ignore_index']),
        '- 图像退化发生在 resize/crop 后的原始 uint8 像素、'
        '`mmlabNormalize` 之前；高斯模糊参数为 kernel=7、sigma=1.5。',
        '- 点云退化发生在 keyframe+sweeps 合并、ToEgo/BEVAug 后，'
        '`PointToMultiViewDepthFusion` 和模型 voxelization 之前，因此'
        ' `gt_depth` 会由退化后的同一份点云重新生成。',
        '- 随机状态由 SHA256(seed, condition, sample token, sensor unit)'
        ' 派生；历史 sweep 固定选择最近 9 帧。每个样本仅构造一次 batch，'
        '两个模型接收同一受损输入。',
        '- 相机失效会将对应相机所有相关时刻的原始像素置零。模型没有'
        '显式 per-camera availability mask；OUP 使用的 annotation 体素'
        ' `mask_camera/mask_lidar` 保持不变，并已在 JSON 中披露。',
        '- 性能下降与保持率均使用本次脚本首先得到的 clean 结果计算，'
        '未硬编码历史 mIoU。',
        '- 表格中的 mIoU 和绝对下降使用官方 evaluator 的百分数/百分点'
        '尺度（例如 54.1400）；保持率按定义另乘 100%。',
        '',
        '## 主表',
        '',
        markdown_table(
            [
                '退化条件', 'Baseline mIoU', 'OUP-Occ mIoU',
                'Baseline下降', 'OUP-Occ下降',
                'Baseline保持率', 'OUP-Occ保持率',
            ],
            [[
                row['condition'],
                fmt(row['baseline_miou'], 4),
                fmt(row['oup_miou'], 4),
                fmt(row['baseline_absolute_drop'], 4),
                fmt(row['oup_absolute_drop'], 4),
                fmt(row['baseline_retention_percent'], 4) + '%',
                fmt(row['oup_retention_percent'], 4) + '%',
            ] for row in rows]),
        '',
        '## 自动分析',
        '',
        category_analysis(
            rows, ['lidar_drop25', 'lidar_drop50'], 'LiDAR 稀疏'),
        category_analysis(
            rows, ['camera_drop1', 'camera_drop3'], '相机缺失'),
        category_analysis(
            rows, ['brightness_0.5', 'gaussian_blur'], '图像质量下降'),
        category_analysis(rows, ['combined'], '联合退化'),
        '',
        '以上只描述真实下降幅度；“更鲁棒”仅在相同 condition 下 OUP 相对'
        '自身 clean 的下降更小或保持率更高时成立，不由模型名称预设。',
        '',
        '## 核验与恢复',
        '',
        '- 每个 condition 的实际点数、丢弃比例、失效相机计数、图像均值'
        '变化及首样本输入 hash 见 JSON 的 `degradation_audit`。',
        '- 每完成一个 condition 会立即保存 `condition_<name>.json`，同时'
        '刷新主 JSON/CSV；`--resume` 只跳过运行签名和样本数均一致的完整'
        '结果。',
        '- 完整逐类 IoU 见 `sensor_degradation_per_class.csv`；运行记录见'
        ' `degradation_runtime_log.txt`。',
    ]
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    configure_runtime(args.seed, args.gpu_id)
    conditions = ordered_conditions(args.conditions)
    paths = resolve_pair_paths(
        args.baseline_config,
        args.baseline_checkpoint,
        args.oup_config,
        args.oup_checkpoint)
    out_dir = absolute_path(args.out_dir)
    mmcv.mkdir_or_exist(out_dir)
    paths['out_dir'] = out_dir
    log_path = os.path.join(out_dir, 'degradation_runtime_log.txt')
    if not args.resume and os.path.isfile(log_path):
        os.remove(log_path)
    logger = RuntimeLogger(log_path)

    baseline_cfg = load_cfg(paths['baseline_config'])
    oup_cfg = load_cfg(paths['oup_config'])
    validate_same_dataset(baseline_cfg, oup_cfg, args.split)
    base_dataset_cfg, base_dataset, data_root, ann_file = (
        build_analysis_dataset(oup_cfg, args.split))
    paths['data_root'] = data_root
    paths['ann_file'] = ann_file
    protocol = official_protocol(
        oup_cfg, base_dataset, args.official_mask)
    baseline_protocol = official_protocol(
        baseline_cfg, base_dataset, args.official_mask)
    if baseline_protocol['num_classes'] != protocol['num_classes']:
        raise ValueError('Baseline/OUP occupancy 类别数不一致')
    limit = sample_limit(args, len(base_dataset))
    protocol.update({
        'processed_samples_per_condition': limit,
        'is_full_val_per_condition': (
            limit == len(base_dataset) == EXPECTED_FULL_VAL_SIZE),
        'online_corruption': True,
        'raw_data_files_modified': False,
        'test_sweep_selection': (
            '固定最近 9 个 sweep（原配置未设置 test_mode，已在内存测试'
            ' pipeline 中纠正）'),
        'image_corruption_stage': (
            '原始像素经过既有 resize/crop 后、Normalize 前'),
        'point_corruption_stage': (
            '多 sweep 合并与坐标变换后、depth projection/voxelization 前'),
        'paired_input_policy': (
            '同一 canonical batch 依次送入 Baseline/OUP'),
        'oup_inference_uses_original_annotation_visibility_masks': True,
        'explicit_per_camera_availability_mask_present': False,
    })
    print_protocol(paths, protocol)

    baseline_model, baseline_checkpoint_audit = build_eval_model(
        baseline_cfg,
        paths['baseline_checkpoint'],
        base_dataset,
        args.gpu_id,
        'Baseline')
    oup_model, oup_checkpoint_audit = build_eval_model(
        oup_cfg,
        paths['oup_checkpoint'],
        base_dataset,
        args.gpu_id,
        'OUP-Occ')
    checkpoint_audit = {
        'baseline': baseline_checkpoint_audit,
        'oup': oup_checkpoint_audit,
    }
    del base_dataset

    completed = {}
    global_start = time.time()
    for condition in conditions:
        # Rebuild the in-memory pipeline per condition. Raw files/configs are
        # never edited and no corruption state leaks between conditions.
        _, dataset, _, _ = build_analysis_dataset(oup_cfg, args.split)
        signature = run_signature(
            condition, args, paths, len(dataset), limit, protocol)
        condition_path = os.path.join(
            out_dir, 'condition_{}.json'.format(condition))
        resumed = (
            load_resumable_condition(condition_path, signature)
            if args.resume else None)
        if resumed is not None:
            logger.log(
                'resume: {} 已完整处理 {} 个样本，跳过'.format(
                    condition, resumed['processed_samples']))
            completed[condition] = resumed
            write_current_outputs(
                out_dir, paths, protocol, checkpoint_audit, completed)
            continue

        spec = install_sensor_corruption(dataset, condition, args.seed)
        loader = build_analysis_loader(
            dataset, args.workers_per_gpu, args.seed)
        logger.log('退化配置核验 {}: {}'.format(condition, spec))
        condition_result = evaluate_condition(
            condition,
            args,
            oup_cfg,
            dataset,
            loader,
            protocol,
            baseline_model,
            oup_model,
            paths,
            limit,
            logger)
        write_json(condition_path, condition_result)
        completed[condition] = condition_result
        _, current_rows = write_current_outputs(
            out_dir, paths, protocol, checkpoint_audit, completed)
        logger.log(
            '完成 {}: Baseline mIoU={:.6f}, OUP mIoU={:.6f}, '
            '即时结果已写入 JSON/CSV'.format(
                condition,
                condition_result['baseline_miou'],
                condition_result['oup_miou']))
        del loader, dataset

    result, rows = write_current_outputs(
        out_dir, paths, protocol, checkpoint_audit, completed)
    make_plots(out_dir, rows)
    write_markdown(
        os.path.join(out_dir, 'sensor_degradation_robustness.md'),
        result,
        rows)

    print('\n传感器退化核心表:')
    print(markdown_table(
        [
            '条件', 'Baseline mIoU', 'OUP mIoU',
            'Baseline下降', 'OUP下降', 'Baseline保持率', 'OUP保持率',
        ],
        [[
            row['condition'],
            fmt(row['baseline_miou'], 4),
            fmt(row['oup_miou'], 4),
            fmt(row['baseline_absolute_drop'], 4),
            fmt(row['oup_absolute_drop'], 4),
            fmt(row['baseline_retention_percent'], 4) + '%',
            fmt(row['oup_retention_percent'], 4) + '%',
        ] for row in rows]))
    finish_message(
        '4.4.3 传感器退化鲁棒性',
        out_dir,
        limit,
        protocol['dataset_samples'],
        time.time() - global_start)


if __name__ == '__main__':
    main()
