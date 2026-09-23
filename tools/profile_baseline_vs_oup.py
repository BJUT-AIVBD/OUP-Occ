#!/usr/bin/env python3
"""Profile the Occ3D Baseline and OUP-Occ on one identical val sample.

The script never trains a model or writes to config/checkpoint files. Data is
loaded and moved to one GPU before profiling, so latency covers model forward
only. FLOPs are reported as a measured lower bound when custom operators are
not supported by MMCV's counter.
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
import warnings
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import DataContainer, collate, scatter
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


DEFAULT_BASELINE_CONFIG = (
    'configs/effocc_fusion_r18_data_scales/'
    'flashocc_fusion_r18_base_100%_seqs.py')
DEFAULT_BASELINE_CHECKPOINT = 'ckpts/effocc_fusion_r18.pth'
DEFAULT_OUP_CONFIG = (
    'configs/oup_occ/oup_occ_fusion_r18_complete_version_mIoU_54_14.py')
DEFAULT_OUP_CHECKPOINT = (
    'work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/'
    'oup_occ_complete_version_mIoU_54_14.pth')

FLOPS_GENERAL_OMISSIONS = [
    'spconv 稀疏卷积及其稀疏张量运算',
    '点云体素化和 BEV pooling 自定义 CUDA 算子',
    '以 torch.nn.functional/张量表达式实现的 grid_sample、softmax、'
    'interpolate、pooling、归约及逐元素运算',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='同一 Occ3D-nuScenes val 样本上的 Baseline/OUP-Occ '
        'FP32 单卡复杂度与效率对比')
    parser.add_argument(
        '--baseline-config', default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument(
        '--baseline-checkpoint', default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument('--baseline-miou', type=float, default=51.5374)
    parser.add_argument('--oup-config', default=DEFAULT_OUP_CONFIG)
    parser.add_argument('--oup-checkpoint', default=DEFAULT_OUP_CHECKPOINT)
    parser.add_argument('--oup-miou', type=float, default=54.14)
    parser.add_argument(
        '--oup-num-queries',
        type=int,
        default=None,
        help='仅在本次 profile 中覆盖 UDCA num_queries，不修改配置文件')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument(
        '--allow-tf32',
        action='store_true',
        help='允许 TF32；默认关闭以执行严格 FP32 前向')
    parser.add_argument(
        '--output',
        default='work_dirs/profile_baseline_vs_oup.json',
        help='JSON 输出路径；传空字符串可禁用文件输出')
    return parser.parse_args()


def resolve_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def require_file(path, role):
    resolved = resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError('{}不存在: {}'.format(role, resolved))
    return resolved


def load_config(path):
    return compat_cfg(Config.fromfile(require_file(path, '配置文件')))


def dataset_identity(cfg):
    val_cfg = cfg.data.val
    return {
        'type': str(val_cfg.get('type', '')),
        'data_root': os.path.normpath(str(val_cfg.get('data_root', ''))),
        'ann_file': os.path.normpath(str(val_cfg.get('ann_file', ''))),
    }


def validate_same_val_split(baseline_cfg, oup_cfg):
    baseline_identity = dataset_identity(baseline_cfg)
    oup_identity = dataset_identity(oup_cfg)
    if baseline_identity != oup_identity:
        raise ValueError(
            'Baseline 与 OUP 配置的 val 数据集不一致，不能保证同样本公平对比: '
            '{} vs {}'.format(baseline_identity, oup_identity))
    return oup_identity


def find_sample_metadata(value):
    """Find useful identifiers without traversing tensor contents."""
    if isinstance(value, DataContainer):
        return find_sample_metadata(value.data)
    if isinstance(value, dict):
        result = {}
        for key in [
                'sample_idx', 'token', 'pts_filename', 'filename',
                'scene_token', 'lidar_token'
        ]:
            if key in value:
                result[key] = value[key]
        if result:
            return result
        for child in value.values():
            result = find_sample_metadata(child)
            if result:
                return result
    elif isinstance(value, (list, tuple)):
        for child in value:
            result = find_sample_metadata(child)
            if result:
                return result
    return {}


def load_one_gpu_sample(dataset, sample_index, gpu_id):
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            'sample-index={} 超出 val 范围 [0, {})'.format(
                sample_index, len(dataset)))

    sample = dataset[sample_index]
    if sample is None:
        raise RuntimeError(
            'val 样本 {} 经测试管线处理后为空'.format(sample_index))
    cpu_batch = collate([sample], samples_per_gpu=1)
    sample_metadata = find_sample_metadata(cpu_batch)

    # This is deliberately outside every measured region. Reusing this exact
    # object also ensures that both models see identical tensors and metadata.
    with torch.cuda.device(gpu_id):
        gpu_batch = scatter(cpu_batch, [gpu_id])[0]
    torch.cuda.synchronize(gpu_id)
    return gpu_batch, sample_metadata, len(dataset)


def disable_pretrained_loading(model_cfg):
    """Avoid unrelated initialization I/O before loading the full checkpoint."""
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg['pretrained'] = None
    img_backbone = model_cfg.get('img_backbone')
    if isinstance(img_backbone, dict):
        img_backbone['pretrained'] = None
    model_cfg['train_cfg'] = None
    return model_cfg


def build_fp32_model(cfg, checkpoint_path, dataset, gpu_id):
    model_cfg = disable_pretrained_loading(cfg.model)
    model = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(
        model,
        require_file(checkpoint_path, 'checkpoint'),
        map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    # Do not call wrap_fp16_model even when a config contains an fp16 section.
    # Resetting fp16_enabled protects against modules that set it themselves.
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    model = model.float().cuda(gpu_id)
    model.eval()
    return model


def forward_raw_logits(model, gpu_batch):
    """Run network inference without CPU formatting or result serialization."""
    return model(
        return_loss=True,
        return_result=True,
        **gpu_batch)


def profile_runtime(model, gpu_batch, gpu_id, warmup, iterations):
    latencies_ms = []
    autocast = torch.cuda.amp.autocast

    with torch.no_grad(), autocast(enabled=False):
        for _ in range(warmup):
            output = forward_raw_logits(model, gpu_batch)
            del output
        torch.cuda.synchronize(gpu_id)

        # The reset is intentionally immediately before measured forwards.
        torch.cuda.reset_peak_memory_stats(gpu_id)
        for _ in range(iterations):
            torch.cuda.synchronize(gpu_id)
            start = time.perf_counter()
            output = forward_raw_logits(model, gpu_batch)
            torch.cuda.synchronize(gpu_id)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            del output

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    variance = sum(
        (latency - mean_ms) ** 2 for latency in latencies_ms
    ) / len(latencies_ms)
    return {
        'latency_mean_ms': mean_ms,
        'latency_std_ms': math.sqrt(variance),
        'latency_std_ddof': 0,
        'fps': 1000.0 / mean_ms,
        'peak_memory_allocated_mib':
            torch.cuda.max_memory_allocated(gpu_id) / (1024.0 ** 2),
        'peak_memory_reserved_mib':
            torch.cuda.max_memory_reserved(gpu_id) / (1024.0 ** 2),
    }


def qualified_type_name(module):
    module_type = type(module)
    return '{}.{}'.format(module_type.__module__, module_type.__name__)


def leaf_modules(model):
    for module in model.modules():
        if not any(True for _ in module.children()):
            yield module


def track_executed_leaf_modules(model):
    executed = Counter()
    handles = []

    def hook(module, _inputs, _output):
        executed[qualified_type_name(module)] += 1

    for module in leaf_modules(model):
        handles.append(module.register_forward_hook(hook))
    return executed, handles


def remove_handles(handles):
    for handle in handles:
        handle.remove()


def fallback_dense_hook_flops(model, gpu_batch):
    """Count dense Conv/ConvTranspose/Linear when MMCV counting fails."""
    total = [0]
    executed, tracker_handles = track_executed_leaf_modules(model)
    flop_handles = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    deconv_types = (
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

    def conv_hook(module, _inputs, output):
        kernel_ops = math.prod(module.kernel_size)
        kernel_ops *= module.in_channels // module.groups
        flops = output.numel() * kernel_ops
        if module.bias is not None:
            flops += output.numel()
        total[0] += int(flops)

    def deconv_hook(module, inputs, output):
        input_tensor = inputs[0]
        spatial_positions = math.prod(input_tensor.shape[2:])
        flops = (
            input_tensor.shape[0]
            * spatial_positions
            * module.in_channels
            * (module.out_channels // module.groups)
            * math.prod(module.kernel_size))
        if module.bias is not None:
            flops += output.numel()
        total[0] += int(flops)

    def linear_hook(module, _inputs, output):
        flops = output.numel() * module.in_features
        if module.bias is not None:
            flops += output.numel()
        total[0] += int(flops)

    for module in model.modules():
        if isinstance(module, conv_types):
            flop_handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, deconv_types):
            flop_handles.append(module.register_forward_hook(deconv_hook))
        elif isinstance(module, nn.Linear):
            flop_handles.append(module.register_forward_hook(linear_hook))

    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            output = forward_raw_logits(model, gpu_batch)
            torch.cuda.synchronize()
            del output
    finally:
        remove_handles(flop_handles)
        remove_handles(tracker_handles)

    supported_names = {
        qualified_type_name(module)
        for module in model.modules()
        if isinstance(module, conv_types + deconv_types + (nn.Linear, ))
    }
    unsupported = sorted(
        name for name in executed if name not in supported_names)
    return float(total[0]), unsupported


def measure_flops(model, gpu_batch):
    """Run MMCV's module-hook counter on the real multimodal val sample."""
    method = 'mmcv_runtime_hooks'
    counter_warnings = []
    try:
        from mmcv.cnn.utils.flops_counter import (
            add_flops_counting_methods,
            get_modules_mapping,
        )

        supported_types = set(get_modules_mapping().keys())
        executed_types = {}
        tracker_handles = []

        def track_type(module, _inputs, _output):
            executed_types[qualified_type_name(module)] = type(module)

        for module in leaf_modules(model):
            tracker_handles.append(module.register_forward_hook(track_type))

        counter_model = add_flops_counting_methods(model)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                counter_model.start_flops_count()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    output = forward_raw_logits(counter_model, gpu_batch)
                    torch.cuda.synchronize()
                    del output
                flops, _ = counter_model.compute_average_flops_cost()
                counter_warnings.extend(str(item.message) for item in caught)
        finally:
            counter_model.stop_flops_count()
            remove_handles(tracker_handles)

        unsupported = sorted(
            name for name, module_type in executed_types.items()
            if module_type not in supported_types)
        return {
            'flops': float(flops),
            'gflops': float(flops) / 1e9,
            'method': method,
            'is_complete': False,
            'multiply_add_convention': 'one',
            'unsupported_executed_leaf_modules': unsupported,
            'counter_warnings': counter_warnings,
            'error': None,
        }
    except Exception as mmcv_error:
        try:
            flops, unsupported = fallback_dense_hook_flops(
                model, gpu_batch)
            return {
                'flops': flops,
                'gflops': flops / 1e9,
                'method': 'dense_conv_linear_runtime_hooks_fallback',
                'is_complete': False,
                'multiply_add_convention': 'one',
                'unsupported_executed_leaf_modules': unsupported,
                'counter_warnings': counter_warnings,
                'error': 'MMCV 计数失败，已回退至主要算子 hook: {}'.format(
                    repr(mmcv_error)),
            }
        except Exception as fallback_error:
            return {
                'flops': None,
                'gflops': None,
                'method': 'failed',
                'is_complete': False,
                'multiply_add_convention': 'one',
                'unsupported_executed_leaf_modules': [],
                'counter_warnings': counter_warnings,
                'error': 'MMCV 与回退 hook 均失败: {}; {}'.format(
                    repr(mmcv_error), repr(fallback_error)),
            }


def profile_model(
        name, cfg, checkpoint, miou, dataset, gpu_batch, gpu_id, warmup,
        iterations):
    model = build_fp32_model(cfg, checkpoint, dataset, gpu_id)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)

    runtime = profile_runtime(
        model, gpu_batch, gpu_id, warmup, iterations)
    peak_allocated = runtime['peak_memory_allocated_mib']
    peak_reserved = runtime['peak_memory_reserved_mib']

    # FLOPs are measured after latency/memory, so profiling hooks cannot affect
    # the timed result or its peak-memory statistic.
    flops = measure_flops(model, gpu_batch)
    result = {
        'name': name,
        'miou': float(miou),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'total_parameters_m': total_params / 1e6,
        'trainable_parameters_m': trainable_params / 1e6,
        **runtime,
        'flops': flops,
    }
    result['peak_memory_allocated_mib'] = peak_allocated
    result['peak_memory_reserved_mib'] = peak_reserved

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    return result


def optional_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def build_comparison(baseline, oup):
    return {
        'miou_gain': oup['miou'] - baseline['miou'],
        'total_parameters_ratio': optional_ratio(
            oup['total_parameters'], baseline['total_parameters']),
        'trainable_parameters_ratio': optional_ratio(
            oup['trainable_parameters'], baseline['trainable_parameters']),
        'counted_flops_ratio': optional_ratio(
            oup['flops']['flops'], baseline['flops']['flops']),
        'peak_memory_ratio': optional_ratio(
            oup['peak_memory_allocated_mib'],
            baseline['peak_memory_allocated_mib']),
        'latency_ratio': optional_ratio(
            oup['latency_mean_ms'], baseline['latency_mean_ms']),
        'fps_ratio': optional_ratio(oup['fps'], baseline['fps']),
    }


def format_number(value, digits=3):
    if value is None:
        return 'N/A'
    return ('{:.%df}' % digits).format(value)


def print_results(results, comparison):
    headers = [
        'Model', 'mIoU', 'Params(M)', 'Trainable(M)', 'Counted GFLOPs*',
        'Peak Mem(MiB)', 'Latency(ms)', 'Std(ms)', 'FPS'
    ]
    rows = []
    for result in results:
        rows.append([
            result['name'],
            format_number(result['miou'], 4),
            format_number(result['total_parameters_m']),
            format_number(result['trainable_parameters_m']),
            format_number(result['flops']['gflops']),
            format_number(result['peak_memory_allocated_mib'], 2),
            format_number(result['latency_mean_ms'], 3),
            format_number(result['latency_std_ms'], 3),
            format_number(result['fps'], 3),
        ])
    widths = [
        max(len(str(item)) for item in [header] + [row[index] for row in rows])
        for index, header in enumerate(headers)
    ]

    print('\nBaseline 与 OUP-Occ 单样本对比（FP32, batch size=1）')
    print('  '.join(
        header.ljust(widths[index])
        for index, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(
            str(item).ljust(widths[index])
            for index, item in enumerate(row)))

    print('\nOUP-Occ 相对 Baseline:')
    print('  mIoU: {:+.4f}'.format(comparison['miou_gain']))
    for key, label in [
            ('total_parameters_ratio', '总参数量'),
            ('trainable_parameters_ratio', '可训练参数量'),
            ('counted_flops_ratio', '已统计 FLOPs'),
            ('peak_memory_ratio', '峰值显存'),
            ('latency_ratio', '平均延迟'),
            ('fps_ratio', 'FPS')
    ]:
        ratio = comparison[key]
        text = 'N/A' if ratio is None else '{:.3f}x'.format(ratio)
        print('  {}: {}'.format(label, text))

    print('\n* Counted GFLOPs 是 MMCV/运行时 hook 实际覆盖算子的下界，'
          '乘加按 1 FLOP 计。')
    print('  固定未统计类别: {}'.format('；'.join(FLOPS_GENERAL_OMISSIONS)))
    for result in results:
        unsupported = result['flops']['unsupported_executed_leaf_modules']
        print('  {} 未支持的已执行叶模块: {}'.format(
            result['name'],
            '、'.join(unsupported) if unsupported else '未检测到'))
        if result['flops']['error']:
            print('  {} FLOPs 计数提示: {}'.format(
                result['name'], result['flops']['error']))


def main():
    args = parse_args()
    if args.warmup < 0:
        raise ValueError('--warmup 必须大于等于 0')
    if args.iterations <= 0:
        raise ValueError('--iterations 必须大于 0')
    if not torch.cuda.is_available():
        raise RuntimeError('该脚本要求单张 CUDA GPU')

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline_cfg = load_config(args.baseline_config)
    oup_cfg = load_config(args.oup_config)
    if args.oup_num_queries is not None:
        if args.oup_num_queries <= 0:
            raise ValueError('--oup-num-queries 必须大于 0')
        udca_cfg = oup_cfg.model.get('deformable_cross_attention')
        if udca_cfg is None:
            raise KeyError('OUP 配置中不存在 deformable_cross_attention')
        udca_cfg.num_queries = args.oup_num_queries
        udca_cfg.topk_ratio = None
    val_identity = validate_same_val_split(baseline_cfg, oup_cfg)

    dataset_cfg = copy.deepcopy(oup_cfg.data.val)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    gpu_batch, sample_metadata, val_size = load_one_gpu_sample(
        dataset, args.sample_index, args.gpu_id)

    print('设备: {} (cuda:{})'.format(
        torch.cuda.get_device_name(args.gpu_id), args.gpu_id))
    print('精度: FP32, TF32={}'.format(args.allow_tf32))
    print('val 样本: index={}/{} metadata={}'.format(
        args.sample_index, val_size, sample_metadata))
    print('预热/计时: {}/{} 次'.format(args.warmup, args.iterations))

    specs = [
        (
            'Baseline', baseline_cfg, args.baseline_checkpoint,
            args.baseline_miou),
        (
            'OUP-Occ' if args.oup_num_queries is None else
            'OUP-Occ-K{}'.format(args.oup_num_queries),
            oup_cfg,
            args.oup_checkpoint,
            args.oup_miou),
    ]
    results = []
    for name, cfg, checkpoint, miou in specs:
        print('\n正在分析 {}...'.format(name))
        results.append(profile_model(
            name=name,
            cfg=cfg,
            checkpoint=checkpoint,
            miou=miou,
            dataset=dataset,
            gpu_batch=gpu_batch,
            gpu_id=args.gpu_id,
            warmup=args.warmup,
            iterations=args.iterations))

    comparison = build_comparison(results[0], results[1])
    report = {
        'environment': {
            'device': torch.cuda.get_device_name(args.gpu_id),
            'gpu_id': args.gpu_id,
            'torch_version': torch.__version__,
            'precision': 'FP32',
            'tf32_enabled': args.allow_tf32,
            'batch_size': 1,
            'warmup_iterations': args.warmup,
            'timed_iterations': args.iterations,
            'latency_scope': (
                'raw model forward only; excludes data loading, host-to-device '
                'transfer, CPU result formatting, and result saving'),
            'forward_mode': 'eval_no_grad_return_result_raw_logits',
            'peak_memory_metric': 'torch.cuda.max_memory_allocated',
            'oup_num_queries': args.oup_num_queries,
        },
        'sample': {
            'split': 'val',
            'index': args.sample_index,
            'val_size': val_size,
            'metadata': sample_metadata,
            'dataset': val_identity,
            'shared_gpu_batch': True,
        },
        'flops_notes': {
            'is_lower_bound': True,
            'general_omissions': FLOPS_GENERAL_OMISSIONS,
            'warning': (
                '未覆盖算子未被估算或伪造；模型间仅可直接比较相同计数口径下'
                '的已统计 FLOPs。'),
        },
        'results': results,
        'comparison_oup_over_baseline': comparison,
    }

    print_results(results, comparison)
    if args.output:
        output_path = resolve_path(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print('\nJSON 结果已保存到: {}'.format(output_path))


if __name__ == '__main__':
    main()
