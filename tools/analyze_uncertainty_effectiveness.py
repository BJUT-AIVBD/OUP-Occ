#!/usr/bin/env python3
"""Evaluate whether native OUP-Occ uncertainty identifies prediction errors."""

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
import torch  # noqa: E402

from analysis_4_4_common import (  # noqa: E402
    DEFAULT_OUP_CHECKPOINT,
    DEFAULT_OUP_CONFIG,
    EXPECTED_FULL_VAL_SIZE,
    VISIBILITY_REGIONS,
    absolute_path,
    build_analysis_dataset,
    build_analysis_loader,
    build_eval_model,
    configure_runtime,
    extract_gt_masks,
    finish_message,
    fmt,
    infer_model,
    load_cfg,
    markdown_table,
    official_protocol,
    output_structure,
    predict_from_output,
    print_protocol,
    resolve_oup_paths,
    sample_limit,
    sample_token,
    visibility_masks,
    write_csv,
    write_json,
)
from mmdet3d.models.occ_uncertainty import (  # noqa: E402
    align_tensor_to_bxyz,
)


DEFAULT_OUT_DIR = (
    'work_dirs/analysis_4_4/4_4_1_uncertainty_effectiveness')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            '4.4.1 不确定性估计有效性：使用 OUP-Occ 原生 coarse/final '
            '预测和 OUE 不确定性，在 Occ3D-nuScenes val 上评测'))
    parser.add_argument('--config', default=DEFAULT_OUP_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_OUP_CHECKPOINT)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument(
        '--official-mask',
        default='camera',
        choices=['camera', 'lidar', 'both', 'union', 'all'])
    parser.add_argument('--full-val', action='store_true')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='smoke test 样本数；--full-val 时忽略')
    parser.add_argument('--quantile-bins', type=int, default=10)
    parser.add_argument('--topk', type=float, nargs='+', default=[5, 10, 20])
    parser.add_argument(
        '--histogram-bins',
        type=int,
        default=65536,
        help='流式 AUROC/AUPR/分位统计的 [0,1] 直方图精度')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--log-interval', type=int, default=50)
    return parser.parse_args()


class UncertaintyHistogram:
    """Streaming bounded-memory statistics over all valid voxels."""

    def __init__(self, bins):
        self.bins = int(bins)
        self.count = np.zeros(self.bins, dtype=np.int64)
        self.sum_u = np.zeros(self.bins, dtype=np.float64)
        self.final_errors = np.zeros(self.bins, dtype=np.float64)
        self.coarse_errors = np.zeros(self.bins, dtype=np.float64)
        self.min_u = float('inf')
        self.max_u = float('-inf')
        self.clipped_low = 0
        self.clipped_high = 0

    def update(self, uncertainty, coarse_error, final_error):
        uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        coarse_error = np.asarray(coarse_error, dtype=np.float64).reshape(-1)
        final_error = np.asarray(final_error, dtype=np.float64).reshape(-1)
        finite = np.isfinite(uncertainty)
        if not finite.all():
            raise RuntimeError(
                'U_total 在有效体素中包含 {} 个 NaN/Inf，无法评测'.format(
                    int((~finite).sum())))
        if uncertainty.size == 0:
            return
        self.min_u = min(self.min_u, float(uncertainty.min()))
        self.max_u = max(self.max_u, float(uncertainty.max()))
        self.clipped_low += int((uncertainty < 0.0).sum())
        self.clipped_high += int((uncertainty > 1.0).sum())
        clipped = np.clip(uncertainty, 0.0, 1.0)
        indices = np.minimum(
            (clipped * self.bins).astype(np.int64), self.bins - 1)
        self.count += np.bincount(indices, minlength=self.bins)
        self.sum_u += np.bincount(
            indices, weights=uncertainty, minlength=self.bins)
        self.coarse_errors += np.bincount(
            indices, weights=coarse_error, minlength=self.bins)
        self.final_errors += np.bincount(
            indices, weights=final_error, minlength=self.bins)

    @property
    def total(self):
        return int(self.count.sum())

    def slice_by_rank(self, start, stop):
        """Aggregate an ascending uncertainty rank interval [start, stop)."""
        start = float(max(0.0, start))
        stop = float(min(float(self.total), stop))
        result = {
            'voxel_count': max(0.0, stop - start),
            'uncertainty_sum': 0.0,
            'coarse_error_count': 0.0,
            'final_error_count': 0.0,
            'uncertainty_min': float('nan'),
            'uncertainty_max': float('nan'),
        }
        if stop <= start:
            return result
        cumulative = 0.0
        first_bin = None
        last_bin = None
        for index in np.flatnonzero(self.count):
            bin_count = float(self.count[index])
            left = cumulative
            right = cumulative + bin_count
            overlap = max(0.0, min(stop, right) - max(start, left))
            cumulative = right
            if overlap <= 0.0:
                if left >= stop:
                    break
                continue
            fraction = overlap / bin_count
            result['uncertainty_sum'] += self.sum_u[index] * fraction
            result['coarse_error_count'] += (
                self.coarse_errors[index] * fraction)
            result['final_error_count'] += (
                self.final_errors[index] * fraction)
            first_bin = index if first_bin is None else first_bin
            last_bin = index
            if right >= stop:
                break
        if first_bin is not None:
            result['uncertainty_min'] = first_bin / self.bins
            result['uncertainty_max'] = (last_bin + 1) / self.bins
        count = result['voxel_count']
        result['mean_uncertainty'] = (
            result['uncertainty_sum'] / count if count else float('nan'))
        result['coarse_error_rate'] = (
            result['coarse_error_count'] / count
            if count else float('nan'))
        result['final_error_rate'] = (
            result['final_error_count'] / count
            if count else float('nan'))
        return result

    def classification_metrics(self):
        positives = float(self.final_errors.sum())
        total = float(self.total)
        negatives = total - positives
        if positives <= 0 or negatives <= 0:
            return {
                'auroc': float('nan'),
                'aupr': float('nan'),
                'error_prevalence': positives / total if total else float('nan'),
            }
        order = np.arange(self.bins - 1, -1, -1)
        tp = np.cumsum(self.final_errors[order])
        fp = np.cumsum(self.count[order] - self.final_errors[order])
        tpr = np.concatenate([[0.0], tp / positives])
        fpr = np.concatenate([[0.0], fp / negatives])
        auroc = float(np.trapz(tpr, fpr))
        precision = np.divide(
            tp,
            tp + fp,
            out=np.ones_like(tp, dtype=np.float64),
            where=(tp + fp) > 0)
        recall = tp / positives
        recall_previous = np.concatenate([[0.0], recall[:-1]])
        aupr = float(np.sum((recall - recall_previous) * precision))
        return {
            'auroc': auroc,
            'aupr': aupr,
            'error_prevalence': positives / total,
        }

    def spearman_binary_error(self):
        """Spearman(U,error), with average ranks for histogram ties."""
        total = float(self.total)
        positives = float(self.final_errors.sum())
        if total <= 1 or positives <= 0 or positives >= total:
            return float('nan')
        cumulative = 0.0
        sum_rank = 0.0
        sum_rank_sq = 0.0
        sum_rank_error = 0.0
        for index in np.flatnonzero(self.count):
            count = float(self.count[index])
            average_rank = cumulative + (count + 1.0) / 2.0
            sum_rank += count * average_rank
            sum_rank_sq += count * average_rank ** 2
            sum_rank_error += (
                float(self.final_errors[index]) * average_rank)
            cumulative += count
        mean_rank = sum_rank / total
        mean_error = positives / total
        covariance = sum_rank_error / total - mean_rank * mean_error
        variance_rank = sum_rank_sq / total - mean_rank ** 2
        variance_error = mean_error * (1.0 - mean_error)
        denominator = math.sqrt(max(0.0, variance_rank * variance_error))
        return covariance / denominator if denominator > 0 else float('nan')


def component_accumulator():
    return {
        name: {
            'correct_sum': 0.0,
            'correct_count': 0,
            'error_sum': 0.0,
            'error_count': 0,
        }
        for name in ['U_total', 'U_sem', 'U_conf', 'U_occ']
    }


def update_components(store, name, values, correct, error):
    if values is None:
        return
    values = np.asarray(values, dtype=np.float64)
    store[name]['correct_sum'] += float(values[correct].sum())
    store[name]['correct_count'] += int(correct.sum())
    store[name]['error_sum'] += float(values[error].sum())
    store[name]['error_count'] += int(error.sum())


def component_summary(store):
    result = {}
    for name, values in store.items():
        result[name] = {
            'correct_mean': (
                values['correct_sum'] / values['correct_count']
                if values['correct_count'] else float('nan')),
            'error_mean': (
                values['error_sum'] / values['error_count']
                if values['error_count'] else float('nan')),
            'correct_count': values['correct_count'],
            'error_count': values['error_count'],
        }
    return result


def optional_uncertainty(uncertainty_dict, key, spatial_shape):
    value = uncertainty_dict.get(key)
    if value is None:
        return None
    aligned = align_tensor_to_bxyz(
        value, target_shape=spatial_shape, name=key)
    return aligned[0].detach().float().cpu().numpy()


def risk_coverage_rows(histogram):
    rows = []
    coverages = np.arange(0.01, 1.001, 0.01)
    for coverage in coverages:
        retained = max(1.0, round(histogram.total * float(coverage)))
        stats = histogram.slice_by_rank(0.0, retained)
        rows.append({
            'coverage': float(coverage),
            'retained_voxels': int(round(stats['voxel_count'])),
            'coarse_risk': stats['coarse_error_rate'],
            'final_risk': stats['final_error_rate'],
        })
    x = np.asarray([row['coverage'] for row in rows])
    coarse = np.asarray([row['coarse_risk'] for row in rows])
    final = np.asarray([row['final_risk'] for row in rows])
    # Risk below 1% is approximated by the first retained interval.
    x_integral = np.concatenate([[0.0], x])
    coarse_integral = np.concatenate([[coarse[0]], coarse])
    final_integral = np.concatenate([[final[0]], final])
    return rows, {
        'coarse_aurc': float(np.trapz(coarse_integral, x_integral)),
        'final_aurc': float(np.trapz(final_integral, x_integral)),
    }


def quantile_rows(histogram, bins):
    rows = []
    for index in range(bins):
        start = histogram.total * index / bins
        stop = histogram.total * (index + 1) / bins
        stats = histogram.slice_by_rank(start, stop)
        coarse = stats['coarse_error_rate']
        final = stats['final_error_rate']
        absolute = coarse - final
        rows.append({
            'quantile': index + 1,
            'percentile_start': 100.0 * index / bins,
            'percentile_end': 100.0 * (index + 1) / bins,
            'uncertainty_min': stats['uncertainty_min'],
            'uncertainty_max': stats['uncertainty_max'],
            'mean_uncertainty': stats['mean_uncertainty'],
            'voxel_count': int(round(stats['voxel_count'])),
            'coarse_error_rate': coarse,
            'final_error_rate': final,
            'absolute_error_rate_drop': absolute,
            'relative_error_rate_drop': (
                absolute / coarse if coarse > 0 else float('nan')),
        })
    return rows


def topk_rows(histogram, topk):
    rows = []
    all_stats = histogram.slice_by_rank(0, histogram.total)
    specifications = [('all', 100.0)]
    specifications.extend([
        ('top_{:g}%'.format(value), float(value)) for value in topk])
    for name, percent in specifications:
        if name == 'all':
            stats = all_stats
        else:
            count = max(1, int(math.ceil(histogram.total * percent / 100.0)))
            stats = histogram.slice_by_rank(
                histogram.total - count, histogram.total)
        coarse = stats['coarse_error_rate']
        final = stats['final_error_rate']
        absolute = coarse - final
        rows.append({
            'region': name,
            'top_percent': percent,
            'voxel_count': int(round(stats['voxel_count'])),
            'coarse_error_rate': coarse,
            'final_error_rate': final,
            'absolute_error_rate_drop': absolute,
            'relative_error_rate_drop': (
                absolute / coarse if coarse > 0 else float('nan')),
        })
    return rows


def make_plots(out_dir, histogram, quantiles, risk_rows):
    centers = (np.arange(histogram.bins) + 0.5) / histogram.bins
    correct_count = histogram.count - histogram.final_errors
    error_count = histogram.final_errors
    fig, ax = plt.subplots(figsize=(8, 5))
    stride = max(1, histogram.bins // 1024)
    plot_x = centers[::stride]
    correct_plot = np.add.reduceat(correct_count, np.arange(
        0, histogram.bins, stride))
    error_plot = np.add.reduceat(error_count, np.arange(
        0, histogram.bins, stride))
    if correct_plot.sum() > 0:
        ax.plot(plot_x[:len(correct_plot)],
                correct_plot / correct_plot.sum(), label='Correct voxels')
    if error_plot.sum() > 0:
        ax.plot(plot_x[:len(error_plot)],
                error_plot / error_plot.sum(), label='Error voxels')
    ax.set_xlabel('U_total')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Uncertainty Distribution: Correct vs Error')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'uncertainty_error_distribution.png'),
        dpi=180)
    plt.close(fig)

    x = [row['quantile'] for row in quantiles]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [100 * row['coarse_error_rate'] for row in quantiles],
        marker='o',
        label='Coarse')
    ax.plot(
        x,
        [100 * row['final_error_rate'] for row in quantiles],
        marker='o',
        label='Final')
    ax.set_xlabel('Equal-count U_total quantile (low to high)')
    ax.set_ylabel('Error rate (%)')
    ax.set_title('Error Rate by Uncertainty Quantile')
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'uncertainty_quantile_error_rate.png'),
        dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    coverage = [100 * row['coverage'] for row in risk_rows]
    ax.plot(
        coverage,
        [100 * row['coarse_risk'] for row in risk_rows],
        label='Coarse')
    ax.plot(
        coverage,
        [100 * row['final_risk'] for row in risk_rows],
        label='Final')
    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Risk / Error Rate (%)')
    ax.set_title('Risk-Coverage Curve')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, 'risk_coverage_curve.png'), dpi=180)
    plt.close(fig)


def write_markdown(path, result, summary_row, topk, visibility, quantiles):
    protocol = result['evaluation_protocol']
    components = result['component_statistics']
    metrics = result['error_detection']
    trend = result['quantile_trend']
    missing = result['intermediate_availability']['missing']
    lines = [
        '# 4.4.1 不确定性估计有效性分析',
        '',
        '本报告由完整模型的真实前向中间量自动生成；未重新训练、未更新'
        '参数，也未修改 checkpoint。',
        '',
        '## 评价协议',
        '',
        '- 数据集：Occ3D-nuScenes validation，实际处理 `{}` / `{}` 个样本。'
        .format(
            protocol['processed_samples'], protocol['dataset_samples']),
        '- 有效体素：`{}`；类别数：`{}`；ignore index：`{}`；'
        'mIoU 口径排除 free 类 `{}`。'.format(
            protocol['mask_tensor_key'],
            protocol['num_classes'],
            protocol['ignore_index'],
            protocol['free_class_index']),
        '- `O_coarse`、`O_final` 和不确定性均直接读取 OUP-Occ '
        '`return_result=True` 输出；U_total 来自 coarse logits 的原生 OUE。',
        '- AUROC/AUPR、Spearman、Top-k、分位和 Risk-Coverage 使用 `{}` 桶'
        '流式直方图统计，最大 U 分辨率为 `{:.8f}`；边界桶按体素比例分配，'
        '避免保存全验证集体素。'.format(
            result['streaming_statistics']['histogram_bins'],
            result['streaming_statistics']['uncertainty_resolution']),
        '- 给定 checkpoint 的 `epoch_meta={}`；完整性核验确认所有模型参数'
        '均有对应权重。'.format(
            result['checkpoint_audit'].get('epoch_meta')),
        '',
        '## 表A：不确定性与真实错误的关系',
        '',
        markdown_table(
            [
                '正确 U_total', '错误 U_total', '错误/正确比',
                'Spearman', 'AUROC', 'AUPR（错误为正类）',
                '错误比例/AUPR随机基线', 'Final AURC',
            ],
            [[
                fmt(summary_row['correct_U_total_mean'], 6),
                fmt(summary_row['error_U_total_mean'], 6),
                fmt(summary_row['error_correct_U_total_ratio'], 4),
                fmt(summary_row['spearman_U_total_error'], 4),
                fmt(summary_row['auroc'], 4),
                fmt(summary_row['aupr'], 4),
                fmt(summary_row['error_prevalence'], 4),
                fmt(summary_row['final_aurc'], 4),
            ]]),
        '',
        '不确定性分量均值：',
        '',
        markdown_table(
            ['分量', '正确体素均值', '错误体素均值'],
            [[
                name,
                fmt(values['correct_mean'], 6),
                fmt(values['error_mean'], 6),
            ] for name, values in components.items()]),
        '',
        '## 表B：高不确定区域 coarse→final 错误率变化',
        '',
        markdown_table(
            [
                '区域', '体素数', 'Coarse错误率', 'Final错误率',
                '绝对下降', '相对下降',
            ],
            [[
                row['region'],
                row['voxel_count'],
                fmt(row['coarse_error_rate'], 4),
                fmt(row['final_error_rate'], 4),
                fmt(row['absolute_error_rate_drop'], 4),
                fmt(row['relative_error_rate_drop'], 4),
            ] for row in topk]),
        '',
        '## 表C：不同可见性区域',
        '',
        markdown_table(
            [
                '区域', '体素数', '平均 U_total',
                'Coarse错误率', 'Final错误率',
            ],
            [[
                VISIBILITY_REGIONS[row['region']],
                row['voxel_count'],
                fmt(row['mean_U_total'], 6),
                fmt(row['coarse_error_rate'], 4),
                fmt(row['final_error_rate'], 4),
            ] for row in visibility]),
        '',
        '## 不确定性分位与 Risk-Coverage',
        '',
        '10 个分位的完整数值见 `uncertainty_quantiles.csv`。'
        'Final 错误率从最低分位的 `{}` 变化到最高分位的 `{}`；'
        '相邻分位非下降次数为 `{}/{}`，Spearman 为 `{}`。'.format(
            fmt(quantiles[0]['final_error_rate'], 4),
            fmt(quantiles[-1]['final_error_rate'], 4),
            trend['nondecreasing_adjacent_pairs'],
            max(0, len(quantiles) - 1),
            fmt(metrics['spearman_U_total_error'], 4)),
        'Coarse AURC=`{}`，Final AURC=`{}`；曲线见 '
        '`risk_coverage_curve.png`。'.format(
            fmt(result['risk_coverage']['coarse_aurc'], 4),
            fmt(result['risk_coverage']['final_aurc'], 4)),
        '',
        '## 自动分析',
        '',
    ]
    ratio = summary_row['error_correct_U_total_ratio']
    if math.isfinite(ratio) and ratio > 1:
        lines.append(
            '- 错误体素的平均 U_total 是正确体素的 `{}` 倍，方向上支持'
            '不确定性对错误的识别能力。'.format(fmt(ratio, 4)))
    else:
        lines.append(
            '- 错误体素的平均 U_total 未高于正确体素，当前结果不支持'
            '“不确定性可有效识别错误”的结论。')
    auroc = metrics['auroc']
    if math.isfinite(auroc):
        lines.append(
            '- 错误检测 AUROC=`{}`、AUPR=`{}`（随机 AUPR 基线=`{}`）；'
            '这些数值应结合错误比例共同解读。'.format(
                fmt(auroc, 4),
                fmt(metrics['aupr'], 4),
                fmt(metrics['error_prevalence'], 4)))
    highest = topk[-1] if topk else None
    if highest is not None:
        lines.append(
            '- `{}` 区域 coarse→final 的绝对错误率变化为 `{}`，'
            '相对变化为 `{}`；正值表示最终阶段降低错误。'.format(
                highest['region'],
                fmt(highest['absolute_error_rate_drop'], 4),
                fmt(highest['relative_error_rate_drop'], 4)))
    empty_regions = [
        VISIBILITY_REGIONS[row['region']]
        for row in visibility if row['voxel_count'] == 0]
    if empty_regions:
        lines.append(
            '- `{}` 在当前官方 mask 下为空，已按要求输出 count=0 和 NaN；'
            '这是评价区域与可见性定义共同造成的，不进行补数。'.format(
                '、'.join(empty_regions)))
    if missing:
        lines.append(
            '- 未获得的中间量：`{}`。JSON 中记录了缺失原因，未用其他公式'
            '伪造。'.format('`、`'.join(missing)))
    else:
        lines.append(
            '- O_coarse、O_final、U_total、U_sem、U_conf、U_occ 和 U_BEV '
            '均从真实前向输出中获得；U_BEV 读取键为 `{}`，全局均值为'
            ' `{}`。'.format(
                result['U_BEV_statistics']['source_key'],
                fmt(result['U_BEV_statistics']['mean'], 6)))
    lines.extend([
        '',
        '## 输出文件',
        '',
        '- `uncertainty_effectiveness.json`：完整机器可读结果',
        '- `uncertainty_summary.csv`：总体统计',
        '- `uncertainty_quantiles.csv`：等数量分位结果',
        '- `topk_uncertainty.csv`：全部及 Top 5/10/20% 结果',
        '- `visibility_uncertainty.csv`：可见性分区',
        '- `risk_coverage.csv`：1%–100% coverage',
        '- `uncertainty_aggregate_histogram.npz`：有界大小的聚合桶数组，'
        '不含全验证集逐体素数据',
        '- 三张 PNG：分布、分位错误率和 Risk-Coverage 曲线',
    ])
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if args.quantile_bins < 2:
        raise ValueError('--quantile-bins 必须至少为 2')
    if args.histogram_bins < 100:
        raise ValueError('--histogram-bins 必须至少为 100')
    if any(value <= 0 or value > 100 for value in args.topk):
        raise ValueError('--topk 必须位于 (0,100]')
    configure_runtime(args.seed, args.gpu_id)

    config_path, checkpoint_path = resolve_oup_paths(
        args.config, args.checkpoint)
    out_dir = absolute_path(args.out_dir)
    mmcv.mkdir_or_exist(out_dir)
    cfg = load_cfg(config_path)
    dataset_cfg, dataset, data_root, ann_file = build_analysis_dataset(
        cfg, args.split)
    loader = build_analysis_loader(
        dataset, args.workers_per_gpu, args.seed)
    protocol = official_protocol(cfg, dataset, args.official_mask)
    paths = {
        'config': config_path,
        'checkpoint': checkpoint_path,
        'data_root': data_root,
        'ann_file': ann_file,
        'out_dir': out_dir,
    }
    print_protocol(paths, protocol)
    model, checkpoint_audit = build_eval_model(
        cfg, checkpoint_path, dataset, args.gpu_id, 'OUP-Occ')

    limit = sample_limit(args, len(dataset))
    histogram = UncertaintyHistogram(args.histogram_bins)
    components = component_accumulator()
    visibility_acc = {
        name: {
            'voxel_count': 0,
            'uncertainty_sum': 0.0,
            'coarse_errors': 0,
            'final_errors': 0,
        }
        for name in VISIBILITY_REGIONS
    }
    availability = {
        name: False for name in [
            'O_coarse', 'O_final', 'U_total', 'U_sem',
            'U_conf', 'U_occ', 'U_BEV',
        ]
    }
    missing_reasons = {}
    u_bev_acc = {
        'sum': 0.0,
        'count': 0,
        'min': float('inf'),
        'max': float('-inf'),
        'source_key': None,
        'first_shape': None,
    }
    first_output_structure = None
    processed = 0
    start_time = time.time()
    print('开始评测: {} / {} 个样本'.format(limit, len(dataset)))

    for index, data_batch in enumerate(loader):
        if index >= limit:
            break
        output = infer_model(model, data_batch)
        if first_output_structure is None:
            first_output_structure = output_structure(output)
            print('首个 forward 返回结构:')
            print(first_output_structure)
        if not isinstance(output, dict):
            raise RuntimeError(
                'OUP forward 未返回字典，不能可靠提取原生中间量')

        num_classes = protocol['num_classes']
        coarse_pred = predict_from_output(
            output, num_classes, prefer='coarse')[0]
        final_pred = predict_from_output(
            output, num_classes, prefer='final')[0]
        availability['O_coarse'] = True
        availability['O_final'] = True
        spatial_shape = tuple(final_pred.shape)
        gt, mask_camera, mask_lidar, valid, _ = extract_gt_masks(
            data_batch,
            spatial_shape,
            num_classes,
            protocol['ignore_index'],
            args.official_mask)

        uncertainty_dict = output.get('uncertainty_dict')
        if not isinstance(uncertainty_dict, dict):
            raise RuntimeError(
                'forward 输出缺少 uncertainty_dict，不能伪造 U_total')
        u_total = optional_uncertainty(
            uncertainty_dict, 'U_total_voxel', spatial_shape)
        if u_total is None:
            raise RuntimeError(
                '真实前向未返回 U_total_voxel，核心评测无法执行')
        availability['U_total'] = True
        optional = {}
        for output_name, key in [
                ('U_sem', 'U_sem_voxel'),
                ('U_conf', 'U_conf_voxel'),
                ('U_occ', 'U_occ_voxel')]:
            optional[output_name] = optional_uncertainty(
                uncertainty_dict, key, spatial_shape)
            availability[output_name] = optional[output_name] is not None
            if optional[output_name] is None:
                missing_reasons[output_name] = (
                    'uncertainty_dict 中不存在 {}'.format(key))
        u_bev_key = (
            'U_total_bev'
            if uncertainty_dict.get('U_total_bev') is not None
            else 'U_total_bev_feat'
            if uncertainty_dict.get('U_total_bev_feat') is not None
            else None)
        availability['U_BEV'] = u_bev_key is not None
        if u_bev_key is not None:
            u_bev = uncertainty_dict[u_bev_key].detach().float().cpu().numpy()
            u_bev_acc['sum'] += float(u_bev.sum())
            u_bev_acc['count'] += int(u_bev.size)
            u_bev_acc['min'] = min(
                u_bev_acc['min'], float(u_bev.min()))
            u_bev_acc['max'] = max(
                u_bev_acc['max'], float(u_bev.max()))
            u_bev_acc['source_key'] = u_bev_key
            if u_bev_acc['first_shape'] is None:
                u_bev_acc['first_shape'] = list(u_bev.shape)
        if not availability['U_BEV']:
            missing_reasons['U_BEV'] = (
                'uncertainty_dict 中不存在 U_total_bev/U_total_bev_feat')

        coarse_error = coarse_pred != gt
        final_error = final_pred != gt
        correct = valid & ~final_error
        error = valid & final_error
        histogram.update(
            u_total[valid], coarse_error[valid], final_error[valid])
        update_components(components, 'U_total', u_total, correct, error)
        for name in ['U_sem', 'U_conf', 'U_occ']:
            update_components(
                components, name, optional[name], correct, error)

        for name, region in visibility_masks(
                mask_camera, mask_lidar).items():
            region_valid = region & valid
            count = int(region_valid.sum())
            visibility_acc[name]['voxel_count'] += count
            visibility_acc[name]['uncertainty_sum'] += float(
                u_total[region_valid].sum())
            visibility_acc[name]['coarse_errors'] += int(
                coarse_error[region_valid].sum())
            visibility_acc[name]['final_errors'] += int(
                final_error[region_valid].sum())
        processed += 1
        del output

        if processed % args.log_interval == 0 or processed == limit:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                '已处理 {}/{}，有效体素={}，平均 {:.3f}s/样本，'
                '最近 token={}'.format(
                    processed,
                    limit,
                    histogram.total,
                    elapsed / processed,
                    sample_token(data_batch, index)))

    if processed == 0 or histogram.total == 0:
        raise RuntimeError('没有处理到有效体素')

    component_stats = component_summary(components)
    detection = histogram.classification_metrics()
    detection['positive_class'] = 'prediction_error'
    detection['score'] = 'U_total'
    detection['spearman_U_total_error'] = (
        histogram.spearman_binary_error())
    risk_rows, aurc = risk_coverage_rows(histogram)
    quantiles = quantile_rows(histogram, args.quantile_bins)
    topk = topk_rows(histogram, sorted(set(args.topk)))
    visibility = []
    for name, values in visibility_acc.items():
        count = values['voxel_count']
        visibility.append({
            'region': name,
            'region_cn': VISIBILITY_REGIONS[name],
            'voxel_count': count,
            'mean_U_total': (
                values['uncertainty_sum'] / count
                if count else float('nan')),
            'coarse_error_rate': (
                values['coarse_errors'] / count
                if count else float('nan')),
            'final_error_rate': (
                values['final_errors'] / count
                if count else float('nan')),
            'empty_reason': (
                None if count else
                '该可见性区域与当前 official mask 的交集为空'),
        })

    u_total_stats = component_stats['U_total']
    ratio = (
        u_total_stats['error_mean'] / u_total_stats['correct_mean']
        if u_total_stats['correct_mean'] > 0 else float('nan'))
    summary_row = {
        'processed_samples': processed,
        'valid_voxels': histogram.total,
        'correct_U_total_mean': u_total_stats['correct_mean'],
        'error_U_total_mean': u_total_stats['error_mean'],
        'error_correct_U_total_ratio': ratio,
        'correct_U_sem_mean': component_stats['U_sem']['correct_mean'],
        'error_U_sem_mean': component_stats['U_sem']['error_mean'],
        'correct_U_conf_mean': component_stats['U_conf']['correct_mean'],
        'error_U_conf_mean': component_stats['U_conf']['error_mean'],
        'correct_U_occ_mean': component_stats['U_occ']['correct_mean'],
        'error_U_occ_mean': component_stats['U_occ']['error_mean'],
        'U_BEV_mean': (
            u_bev_acc['sum'] / u_bev_acc['count']
            if u_bev_acc['count'] else float('nan')),
        'spearman_U_total_error': detection['spearman_U_total_error'],
        'auroc': detection['auroc'],
        'aupr': detection['aupr'],
        'error_prevalence': detection['error_prevalence'],
        'coarse_aurc': aurc['coarse_aurc'],
        'final_aurc': aurc['final_aurc'],
    }
    final_rates = [row['final_error_rate'] for row in quantiles]
    nondecreasing = sum(
        right >= left
        for left, right in zip(final_rates[:-1], final_rates[1:]))
    protocol.update({
        'processed_samples': processed,
        'is_full_val': (
            processed == len(dataset) == EXPECTED_FULL_VAL_SIZE),
        'dataset_config_pipeline': [
            step.get('type') for step in dataset_cfg.pipeline],
    })
    missing = [name for name, value in availability.items() if not value]
    result = {
        'resolved_paths': paths,
        'evaluation_protocol': protocol,
        'checkpoint_audit': checkpoint_audit,
        'forward_output_structure': first_output_structure,
        'intermediate_availability': {
            'available': [
                name for name, value in availability.items() if value],
            'missing': missing,
            'missing_reasons': missing_reasons,
            'RAPR_mapping_note': (
                '仓库无名为 RAPR 的独立模块；按真实实现将 '
                'coarse_occ_logits 视为 refinement 前预测，'
                'output_occ_logits 视为最终预测。'),
        },
        'streaming_statistics': {
            'histogram_bins': histogram.bins,
            'uncertainty_resolution': 1.0 / histogram.bins,
            'uncertainty_observed_min': histogram.min_u,
            'uncertainty_observed_max': histogram.max_u,
            'values_clipped_below_zero': histogram.clipped_low,
            'values_clipped_above_one': histogram.clipped_high,
            'boundary_tie_handling': (
                'Top-k/分位边界桶按桶内体素比例分配'),
            'full_voxel_arrays_saved': False,
        },
        'summary': summary_row,
        'component_statistics': component_stats,
        'U_BEV_statistics': {
            'source_key': u_bev_acc['source_key'],
            'first_shape': u_bev_acc['first_shape'],
            'mean': (
                u_bev_acc['sum'] / u_bev_acc['count']
                if u_bev_acc['count'] else float('nan')),
            'min': (
                u_bev_acc['min']
                if u_bev_acc['count'] else float('nan')),
            'max': (
                u_bev_acc['max']
                if u_bev_acc['count'] else float('nan')),
            'value_count': u_bev_acc['count'],
        },
        'error_detection': detection,
        'topk_uncertainty': topk,
        'uncertainty_quantiles': quantiles,
        'quantile_trend': {
            'nondecreasing_adjacent_pairs': nondecreasing,
            'total_adjacent_pairs': max(0, len(quantiles) - 1),
            'highest_minus_lowest_final_error_rate': (
                final_rates[-1] - final_rates[0]),
        },
        'risk_coverage': {
            'coarse_aurc': aurc['coarse_aurc'],
            'final_aurc': aurc['final_aurc'],
            'rows': risk_rows,
        },
        'visibility_uncertainty': visibility,
    }

    write_json(
        os.path.join(out_dir, 'uncertainty_effectiveness.json'), result)
    write_csv(
        os.path.join(out_dir, 'uncertainty_summary.csv'), [summary_row])
    write_csv(
        os.path.join(out_dir, 'uncertainty_quantiles.csv'), quantiles)
    write_csv(
        os.path.join(out_dir, 'topk_uncertainty.csv'), topk)
    write_csv(
        os.path.join(out_dir, 'visibility_uncertainty.csv'), visibility)
    write_csv(
        os.path.join(out_dir, 'risk_coverage.csv'), risk_rows)
    # These are bounded-size aggregate arrays (a few MiB), never per-voxel
    # validation dumps.
    np.savez_compressed(
        os.path.join(out_dir, 'uncertainty_aggregate_histogram.npz'),
        count=histogram.count,
        sum_u=histogram.sum_u,
        coarse_errors=histogram.coarse_errors,
        final_errors=histogram.final_errors,
        histogram_bins=histogram.bins)
    make_plots(out_dir, histogram, quantiles, risk_rows)
    write_markdown(
        os.path.join(out_dir, 'uncertainty_effectiveness.md'),
        result,
        summary_row,
        topk,
        visibility,
        quantiles)

    print('\n表A核心统计:')
    print(markdown_table(
        ['正确U', '错误U', '比值', 'AUROC', 'AUPR', 'Final AURC'],
        [[
            fmt(summary_row['correct_U_total_mean'], 4),
            fmt(summary_row['error_U_total_mean'], 4),
            fmt(ratio, 4),
            fmt(detection['auroc'], 4),
            fmt(detection['aupr'], 4),
            fmt(aurc['final_aurc'], 4),
        ]]))
    finish_message(
        '4.4.1 不确定性有效性',
        out_dir,
        processed,
        len(dataset),
        time.time() - start_time)


if __name__ == '__main__':
    main()
