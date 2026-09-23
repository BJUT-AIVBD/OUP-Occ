#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from mmcv import Config

from mmdet3d.models import build_model


HEAD_KEYWORDS = (
    'occ_head', 'final_occ_head', 'coarse_occ_head', 'predicter',
    'final_conv'
)


def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint, checkpoint['state_dict']
    return {'meta': {}}, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f'{out_path} exists; pass --overwrite to replace it.')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    target_state = model.state_dict()

    checkpoint, source_state = load_state_dict(args.src)
    filtered = {}
    loaded = []
    skipped = []

    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append((key, 'unexpected_in_target', tuple(value.shape)))
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            reason = 'shape_mismatch'
            if any(token in key for token in HEAD_KEYWORDS):
                reason = 'shape_mismatch_occupancy_head'
            skipped.append((key, reason, tuple(value.shape), tuple(target_value.shape)))
            continue
        filtered[key] = value
        loaded.append(key)

    missing = sorted(set(target_state.keys()) - set(filtered.keys()))
    unexpected = sorted(set(source_state.keys()) - set(target_state.keys()))

    new_checkpoint = dict(
        state_dict=filtered,
        meta=dict(
            source_checkpoint=str(args.src),
            target_config=str(args.config),
            note='Shape-compatible OUP-Occ nuScenes weights for SSCBench-KITTI initialization.'))
    torch.save(new_checkpoint, out_path)

    report_path = Path(args.report) if args.report else out_path.with_name(
        out_path.stem + '_load_report.txt')
    lines = [
        f'Source checkpoint: {args.src}',
        f'Target config: {args.config}',
        f'Output checkpoint: {out_path}',
        f'Loaded keys: {len(loaded)}',
        f'Skipped keys: {len(skipped)}',
        f'Missing target keys after filtering: {len(missing)}',
        f'Unexpected source keys: {len(unexpected)}',
        '',
        '[Loaded keys]',
        *loaded,
        '',
        '[Skipped keys]',
    ]
    for item in skipped:
        lines.append(repr(item))
    lines.extend(['', '[Missing keys]', *missing, '', '[Unexpected keys]', *unexpected])
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved filtered checkpoint to {out_path}')
    print(f'Saved load report to {report_path}')


if __name__ == '__main__':
    main()
