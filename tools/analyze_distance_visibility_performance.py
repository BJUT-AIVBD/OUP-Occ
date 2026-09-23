#!/usr/bin/env python3
"""Compare Baseline and OUP-Occ by physical distance and visibility."""

import argparse
import math
import os
import sys
import time
from collections import OrderedDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mmcv  # noqa: E402
import numpy as np  # noqa: E402

from analysis_4_4_common import (  # noqa: E402
    DEFAULT_BASELINE_CHECKPOINT,
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_OUP_CHECKPOINT,
    DEFAULT_OUP_CONFIG,
    EXPECTED_FULL_VAL_SIZE,
    VISIBILITY_REGIONS,
    absolute_path,
    build_analysis_dataset,
    build_analysis_loader,
    build_eval_model,
    configure_runtime,
    confusion_matrix,
    distance_bin_specs,
    distance_volume,
    extract_gt_masks,
    finish_message,
    fmt,
    infer_model,
    load_cfg,
    markdown_table,
    metrics_from_hist,
    official_protocol,
    output_structure,
    parse_distance_edges,
    predict_from_output,
    print_protocol,
    resolve_pair_paths,
    sample_limit,
    sample_token,
    validate_same_dataset,
    visibility_masks,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = 'work_dirs/analysis_4_4/4_4_2_distance_visibility'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            '4.4.2 在相同 Occ3D val 样本、官方 mask 和类别定义下，'
            '比较 Baseline/OUP-Occ 的距离与可见性区域性能'))
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
        '--distance-bins',
        nargs='+',
        default=['0', '15', '30', '45', 'inf'])
    parser.add_argument('--full-val', action='store_true')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='smoke test 样本数；--full-val 时忽略')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--log-interval', type=int, default=50)
    return parser.parse_args()


def new_region(num_classes):
    return {
        'voxel_count': 0,
        'baseline_errors': 0,
        'oup_errors': 0,
        'coarse_errors': 0,
        'coarse_available': True,
        'baseline_hist': np.zeros(
            (num_classes, num_classes), dtype=np.int64),
        'oup_hist': np.zeros(
            (num_classes, num_classes), dtype=np.int64),
    }


def update_region(
        region, baseline_pred, oup_pred, coarse_pred, gt, valid, num_classes):
    count = int(valid.sum())
    region['voxel_count'] += count
    region['baseline_errors'] += int(
        ((baseline_pred != gt) & valid).sum())
    region['oup_errors'] += int(((oup_pred != gt) & valid).sum())
    if coarse_pred is None:
        region['coarse_available'] = False
    else:
        region['coarse_errors'] += int(
            ((coarse_pred != gt) & valid).sum())
    region['baseline_hist'] += confusion_matrix(
        baseline_pred, gt, valid, num_classes)
    region['oup_hist'] += confusion_matrix(
        oup_pred, gt, valid, num_classes)


def finalize_regions(regions, protocol):
    summaries = []
    per_class_rows = []
    total_valid = sum(
        value['voxel_count'] for value in regions.values())
    class_names = protocol['class_names']
    free_class = protocol['free_class_index']
    for name, values in regions.items():
        baseline = metrics_from_hist(values['baseline_hist'], free_class)
        oup = metrics_from_hist(values['oup_hist'], free_class)
        count = values['voxel_count']
        baseline_error = (
            values['baseline_errors'] / count
            if count else float('nan'))
        oup_error = (
            values['oup_errors'] / count
            if count else float('nan'))
        coarse_error = (
            values['coarse_errors'] / count
            if count and values['coarse_available'] else float('nan'))
        summaries.append({
            'region': name,
            'voxel_count': count,
            'voxel_ratio': (
                count / total_valid if total_valid else float('nan')),
            'baseline_error_rate': baseline_error,
            'oup_error_rate': oup_error,
            'absolute_error_rate_drop': baseline_error - oup_error,
            'baseline_miou': baseline['miou'],
            'oup_miou': oup['miou'],
            'miou_improvement': oup['miou'] - baseline['miou'],
            'oup_coarse_error_rate': coarse_error,
            'oup_final_error_rate': oup_error,
            'oup_coarse_to_final_error_drop': coarse_error - oup_error,
            'empty_reason': (
                None if count else
                '该区域与当前 official mask 的交集为空'),
        })
        for class_index, class_name in enumerate(class_names):
            baseline_iou = baseline['per_class_iou'][class_index]
            oup_iou = oup['per_class_iou'][class_index]
            per_class_rows.append({
                'region': name,
                'class_index': class_index,
                'class_name': class_name,
                'included_in_miou': class_index < free_class,
                'baseline_iou': baseline_iou,
                'oup_iou': oup_iou,
                'iou_improvement': oup_iou - baseline_iou,
                'absent_in_both_gt_and_prediction': (
                    not math.isfinite(float(baseline_iou))
                    and not math.isfinite(float(oup_iou))),
            })
    return summaries, per_class_rows


def make_plots(out_dir, distance_rows, visibility_rows):
    x = np.arange(len(distance_rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        [row['baseline_miou'] for row in distance_rows],
        width,
        label='Baseline')
    ax.bar(
        x + width / 2,
        [row['oup_miou'] for row in distance_rows],
        width,
        label='OUP-Occ')
    ax.set_xticks(x)
    ax.set_xticklabels([row['region'] for row in distance_rows])
    ax.set_ylabel('mIoU (%)')
    ax.set_title('mIoU by Horizontal Distance')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'distance_performance.png'), dpi=180)
    plt.close(fig)

    x = np.arange(len(visibility_rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    baseline = [
        row['baseline_miou']
        if math.isfinite(row['baseline_miou']) else np.nan
        for row in visibility_rows]
    oup = [
        row['oup_miou']
        if math.isfinite(row['oup_miou']) else np.nan
        for row in visibility_rows]
    ax.bar(x - width / 2, baseline, width, label='Baseline')
    ax.bar(x + width / 2, oup, width, label='OUP-Occ')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row['region'] for row in visibility_rows], rotation=12)
    ax.set_ylabel('mIoU (%)')
    ax.set_title('mIoU by Visibility Region')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'visibility_performance.png'), dpi=180)
    plt.close(fig)


def strongest_supported_statement(rows, label):
    nonempty = [
        row for row in rows
        if row['voxel_count'] > 0
        and math.isfinite(row['miou_improvement'])]
    if not nonempty:
        return '- {}没有非空且可计算的区域，无法比较改善幅度。'.format(label)
    best = max(nonempty, key=lambda row: row['miou_improvement'])
    worst = min(nonempty, key=lambda row: row['miou_improvement'])
    return (
        '- {}中，mIoU 提升最大的是 `{}`（`{}`），最小的是 `{}`'
        '（`{}`）；该描述仅陈述本次结果，不预设远距离或困难区域必然'
        '改善。'.format(
            label,
            best['region'],
            fmt(best['miou_improvement'], 4),
            worst['region'],
            fmt(worst['miou_improvement'], 4)))


def write_markdown(path, result, distance_rows, visibility_rows):
    protocol = result['evaluation_protocol']
    lines = [
        '# 4.4.2 不同距离与可见性区域的性能分析',
        '',
        '本报告使用同一个 validation DataLoader，将完全相同的 batch 依次'
        '送入 Baseline 与 OUP-Occ；两模型均为 eval/FP32/no_grad、单 GPU、'
        'batch size=1。',
        '',
        '## 评价协议',
        '',
        '- 数据集：Occ3D-nuScenes validation，实际处理 `{}` / `{}` 个'
        '样本；评价 mask 为 `{}`。'.format(
            protocol['processed_samples'],
            protocol['dataset_samples'],
            protocol['mask_tensor_key']),
        '- 类别数 `{}`，ignore index `{}`，mIoU 对类 0–{} 使用全局'
        '混淆矩阵后 `nanmean`；free 类 {} 不进入平均。'.format(
            protocol['num_classes'],
            protocol['ignore_index'],
            protocol['free_class_index'] - 1,
            protocol['free_class_index']),
        '- 表格中的 mIoU/IoU/提升使用与仓库官方 evaluator 一致的百分数'
        '尺度（例如 54.1400）；错误率使用 0–1 比例。',
        '- 体素中心坐标由 point cloud range `{}` 与预测 grid `{}` 自动'
        '计算；水平距离为 `sqrt(x²+y²)`，未使用体素索引代替米制坐标。'
        .format(
            protocol['point_cloud_range'], protocol['observed_grid_shape']),
        '- absent class 遵循仓库 evaluator：GT 与预测均不存在时 IoU=NaN，'
        '由 `nanmean` 排除，不填 0 或 1。',
        '- 审计说明：当前最终 OUP 配置会在推理中使用 annotation 提供的'
        ' `mask_camera/mask_lidar` 作为可见性先验；Baseline 忽略这些附加'
        '字段。评价 mask 对两者仍完全相同。',
        '',
        '## 表A：不同距离区域',
        '',
        markdown_table(
            [
                '距离', '体素数', 'Baseline mIoU', 'OUP mIoU',
                'mIoU提升', 'Baseline错误率', 'OUP错误率', '错误率下降',
            ],
            [[
                row['region'],
                row['voxel_count'],
                fmt(row['baseline_miou'], 4),
                fmt(row['oup_miou'], 4),
                fmt(row['miou_improvement'], 4),
                fmt(row['baseline_error_rate'], 4),
                fmt(row['oup_error_rate'], 4),
                fmt(row['absolute_error_rate_drop'], 4),
            ] for row in distance_rows]),
        '',
        '## 表B：不同可见性区域',
        '',
        markdown_table(
            [
                '区域', '体素数', '占比', 'Baseline mIoU', 'OUP mIoU',
                'mIoU提升', 'Baseline错误率', 'OUP错误率', '错误率下降',
            ],
            [[
                VISIBILITY_REGIONS[row['region']],
                row['voxel_count'],
                fmt(row['voxel_ratio'], 4),
                fmt(row['baseline_miou'], 4),
                fmt(row['oup_miou'], 4),
                fmt(row['miou_improvement'], 4),
                fmt(row['baseline_error_rate'], 4),
                fmt(row['oup_error_rate'], 4),
                fmt(row['absolute_error_rate_drop'], 4),
            ] for row in visibility_rows]),
        '',
        'OUP-Occ 的 coarse/final 错误率及每类 IoU 分别保存在详细 CSV 中。',
        '',
        '## 自动分析',
        '',
        strongest_supported_statement(distance_rows, '距离分区'),
        strongest_supported_statement(visibility_rows, '可见性分区'),
    ]
    empty = [
        VISIBILITY_REGIONS[row['region']]
        for row in visibility_rows if row['voxel_count'] == 0]
    if empty:
        lines.append(
            '- `{}` 在当前 `{}` 评价区域中为空，已如实保留 count=0 和'
            ' NaN。尤其在官方 camera mask 下，camera 不可见区域通常不会'
            '进入评价，不能据此声称模型在该区域性能更高或更低。'.format(
                '、'.join(empty), protocol['mask_tensor_key']))
    far = next(
        (row for row in distance_rows
         if row.get('is_open_ended', False)),
        None)
    if far is not None and far['voxel_count'] > 0:
        if far['miou_improvement'] > 0:
            conclusion = '提升'
        elif far['miou_improvement'] < 0:
            conclusion = '下降'
        else:
            conclusion = '持平'
        lines.append(
            '- 最远距离分区的 OUP mIoU 相对 Baseline {} `{}`，错误率变化'
            '为 `{}`；是否“远距离改善更明显”应与其他距离分区的提升共同'
            '比较。'.format(
                conclusion,
                fmt(abs(far['miou_improvement']), 4),
                fmt(far['absolute_error_rate_drop'], 4)))
    lines.extend([
        '',
        '## 输出文件',
        '',
        '- `distance_visibility_performance.json`：协议、模型输出结构和完整结果',
        '- `distance_bins_summary.csv` / `distance_bins_per_class.csv`',
        '- `visibility_regions_summary.csv` / `visibility_regions_per_class.csv`',
        '- `distance_performance.png` / `visibility_performance.png`',
    ])
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    edges = parse_distance_edges(args.distance_bins)
    configure_runtime(args.seed, args.gpu_id)
    paths = resolve_pair_paths(
        args.baseline_config,
        args.baseline_checkpoint,
        args.oup_config,
        args.oup_checkpoint)
    out_dir = absolute_path(args.out_dir)
    mmcv.mkdir_or_exist(out_dir)
    paths['out_dir'] = out_dir

    baseline_cfg = load_cfg(paths['baseline_config'])
    oup_cfg = load_cfg(paths['oup_config'])
    validate_same_dataset(baseline_cfg, oup_cfg, args.split)
    dataset_cfg, dataset, data_root, ann_file = build_analysis_dataset(
        oup_cfg, args.split)
    paths['data_root'] = data_root
    paths['ann_file'] = ann_file
    loader = build_analysis_loader(
        dataset, args.workers_per_gpu, args.seed)
    protocol = official_protocol(oup_cfg, dataset, args.official_mask)
    if official_protocol(
            baseline_cfg, dataset, args.official_mask)['num_classes'] != (
                protocol['num_classes']):
        raise ValueError('Baseline/OUP occupancy 类别数不一致')
    print_protocol(paths, protocol)

    baseline_model, baseline_checkpoint_audit = build_eval_model(
        baseline_cfg,
        paths['baseline_checkpoint'],
        dataset,
        args.gpu_id,
        'Baseline')
    oup_model, oup_checkpoint_audit = build_eval_model(
        oup_cfg,
        paths['oup_checkpoint'],
        dataset,
        args.gpu_id,
        'OUP-Occ')

    num_classes = protocol['num_classes']
    distance_regions = OrderedDict(
        (name, new_region(num_classes))
        for name, _, _ in distance_bin_specs(edges))
    visibility_regions = OrderedDict(
        (name, new_region(num_classes)) for name in VISIBILITY_REGIONS)
    first_structures = {}
    distance_grid = None
    limit = sample_limit(args, len(dataset))
    processed = 0
    start_time = time.time()
    print('开始成对评测: {} / {} 个样本'.format(limit, len(dataset)))

    for index, data_batch in enumerate(loader):
        if index >= limit:
            break
        baseline_output = infer_model(baseline_model, data_batch)
        baseline_pred = predict_from_output(
            baseline_output, num_classes, prefer='final')[0]
        if not first_structures:
            first_structures['baseline'] = output_structure(baseline_output)
        del baseline_output

        oup_output = infer_model(oup_model, data_batch)
        oup_pred = predict_from_output(
            oup_output, num_classes, prefer='final')[0]
        try:
            coarse_pred = predict_from_output(
                oup_output, num_classes, prefer='coarse')[0]
        except (KeyError, TypeError, ValueError):
            coarse_pred = None
        if 'oup' not in first_structures:
            first_structures['oup'] = output_structure(oup_output)
            print('首个 forward 返回结构:')
            print(first_structures)
        del oup_output

        if baseline_pred.shape != oup_pred.shape:
            raise ValueError(
                'Baseline/OUP 预测 shape 不一致: {} vs {}'.format(
                    baseline_pred.shape, oup_pred.shape))
        spatial_shape = tuple(oup_pred.shape)
        gt, mask_camera, mask_lidar, valid, _ = extract_gt_masks(
            data_batch,
            spatial_shape,
            num_classes,
            protocol['ignore_index'],
            args.official_mask)
        if distance_grid is None:
            distance_grid = distance_volume(
                spatial_shape, protocol['point_cloud_range'])
            protocol['observed_grid_shape'] = list(spatial_shape)
            protocol['voxel_center_formula'] = (
                'axis_min + (index + 0.5) * '
                '(axis_max-axis_min)/axis_dimension')

        for name, lower, upper in distance_bin_specs(edges):
            region_mask = (
                (distance_grid >= lower) & (distance_grid < upper))
            update_region(
                distance_regions[name],
                baseline_pred,
                oup_pred,
                coarse_pred,
                gt,
                valid & region_mask,
                num_classes)
        for name, region_mask in visibility_masks(
                mask_camera, mask_lidar).items():
            update_region(
                visibility_regions[name],
                baseline_pred,
                oup_pred,
                coarse_pred,
                gt,
                valid & region_mask,
                num_classes)

        processed += 1
        if processed % args.log_interval == 0 or processed == limit:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                '已处理 {}/{}，平均 {:.3f}s/样本，最近 token={}'.format(
                    processed,
                    limit,
                    elapsed / processed,
                    sample_token(data_batch, index)))

    if processed == 0:
        raise RuntimeError('没有处理任何样本')
    distance_rows, distance_per_class = finalize_regions(
        distance_regions, protocol)
    for row, (_, lower, upper) in zip(
            distance_rows, distance_bin_specs(edges)):
        row['lower_bound_m'] = lower
        row['upper_bound_m'] = 'inf' if math.isinf(upper) else upper
        row['is_open_ended'] = math.isinf(upper)
        row['voxel_ratio'] = (
            row['voxel_count']
            / sum(item['voxel_count'] for item in distance_rows))
    visibility_rows, visibility_per_class = finalize_regions(
        visibility_regions, protocol)

    protocol.update({
        'processed_samples': processed,
        'is_full_val': (
            processed == len(dataset) == EXPECTED_FULL_VAL_SIZE),
        'distance_definition': 'sqrt(x^2+y^2) at physical voxel centers',
        'distance_bins_m': [
            'inf' if math.isinf(value) else value for value in edges],
        'dataset_config_pipeline': [
            step.get('type') for step in dataset_cfg.pipeline],
        'paired_input_policy': (
            '同一 DataLoader batch 依次送入两个模型，顺序不变'),
        'oup_inference_uses_annotation_visibility_masks': True,
    })
    result = {
        'resolved_paths': paths,
        'evaluation_protocol': protocol,
        'checkpoint_audit': {
            'baseline': baseline_checkpoint_audit,
            'oup': oup_checkpoint_audit,
        },
        'forward_output_structure': first_structures,
        'distance_bins': distance_rows,
        'distance_bins_per_class': distance_per_class,
        'visibility_regions': visibility_rows,
        'visibility_regions_per_class': visibility_per_class,
    }
    write_json(
        os.path.join(
            out_dir, 'distance_visibility_performance.json'),
        result)
    write_csv(
        os.path.join(out_dir, 'distance_bins_summary.csv'),
        distance_rows)
    write_csv(
        os.path.join(out_dir, 'distance_bins_per_class.csv'),
        distance_per_class)
    write_csv(
        os.path.join(out_dir, 'visibility_regions_summary.csv'),
        visibility_rows)
    write_csv(
        os.path.join(out_dir, 'visibility_regions_per_class.csv'),
        visibility_per_class)
    make_plots(out_dir, distance_rows, visibility_rows)
    write_markdown(
        os.path.join(out_dir, 'distance_visibility_performance.md'),
        result,
        distance_rows,
        visibility_rows)

    print('\n距离区域核心表:')
    print(markdown_table(
        ['距离', '体素数', 'Baseline mIoU', 'OUP mIoU', '提升'],
        [[
            row['region'],
            row['voxel_count'],
            fmt(row['baseline_miou'], 4),
            fmt(row['oup_miou'], 4),
            fmt(row['miou_improvement'], 4),
        ] for row in distance_rows]))
    finish_message(
        '4.4.2 距离与可见性',
        out_dir,
        processed,
        len(dataset),
        time.time() - start_time)


if __name__ == '__main__':
    main()
