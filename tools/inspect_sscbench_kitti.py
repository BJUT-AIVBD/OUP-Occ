#!/usr/bin/env python
import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
LIDAR_EXTS = {'.bin', '.pcd'}
ARRAY_EXTS = {'.npy', '.npz'}
RAW_LABEL_EXTS = {'.label', '.invalid'}
PKL_EXTS = {'.pkl', '.pickle'}
CALIB_EXTS = {'.txt', '.yaml', '.yml', '.json'}
SPLIT_HINTS = {'train', 'val', 'valid', 'test'}
OCC_HINTS = (
    'occ', 'occupancy', 'ssc', 'label', 'labels', 'voxel', 'voxels',
    'semantic', 'semantics'
)


def rel(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_hidden(path):
    return any(part.startswith('.') for part in path.parts)


def walk_dataset(root):
    files = []
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if is_hidden(current.relative_to(root)):
            continue
        dirs.append(current)
        for name in filenames:
            files.append(current / name)
    return dirs, files


def limited_tree(dirs, files, root, max_depth=3, limit=120):
    lines = []
    emitted = 0
    for directory in sorted(dirs):
        depth = len(directory.relative_to(root).parts)
        if depth <= max_depth:
            lines.append(f"[D] {rel(directory, root) or '.'}")
            emitted += 1
            if emitted >= limit:
                lines.append(f"... truncated after {limit} entries")
                return lines
    for file_path in sorted(files):
        depth = len(file_path.relative_to(root).parts)
        if depth <= max_depth:
            lines.append(f"[F] {rel(file_path, root)}")
            emitted += 1
            if emitted >= limit:
                lines.append(f"... truncated after {limit} entries")
                return lines
    return lines


def sample_files(files, predicate, limit):
    result = []
    for path in sorted(files):
        if predicate(path):
            result.append(path)
            if len(result) >= limit:
                break
    return result


def array_stats(arr):
    info = {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
    }
    if arr.size == 0:
        info.update(min=None, max=None)
        return info
    try:
        if arr.size > 5_000_000:
            flat = arr.reshape(-1)
            sample = flat[::max(1, flat.size // 1_000_000)]
            info['sampled'] = True
            info['min'] = safe_scalar(np.nanmin(sample))
            info['max'] = safe_scalar(np.nanmax(sample))
        else:
            info['min'] = safe_scalar(np.nanmin(arr))
            info['max'] = safe_scalar(np.nanmax(arr))
    except Exception as exc:
        info['minmax_error'] = str(exc)
    return info


def safe_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_npz(path):
    result = {'type': 'npz', 'keys': {}}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            result['keys'][key] = array_stats(arr)
    return result


def inspect_npy(path):
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    return {'type': 'npy', 'array': array_stats(arr)}


def summarize_obj(obj, depth=0):
    if depth >= 3:
        return type(obj).__name__
    if isinstance(obj, dict):
        out = {'type': 'dict', 'len': len(obj), 'keys': list(obj.keys())[:30]}
        preview = {}
        for key in list(obj.keys())[:8]:
            preview[str(key)] = summarize_obj(obj[key], depth + 1)
        out['preview'] = preview
        return out
    if isinstance(obj, (list, tuple)):
        return {
            'type': type(obj).__name__,
            'len': len(obj),
            'first': summarize_obj(obj[0], depth + 1) if len(obj) else None,
        }
    if isinstance(obj, np.ndarray):
        return {'type': 'ndarray', **array_stats(obj)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return type(obj).__name__


def inspect_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return {'type': 'pkl', 'summary': summarize_obj(obj)}


def inspect_file(path):
    suffix = path.suffix.lower()
    try:
        if suffix == '.npz':
            return inspect_npz(path)
        if suffix == '.npy':
            return inspect_npy(path)
        if suffix in PKL_EXTS:
            return inspect_pkl(path)
        if suffix == '.label':
            return inspect_raw_binary(path, np.uint16, 'label_uint16')
        if suffix == '.invalid':
            return inspect_raw_binary(path, np.uint8, 'invalid_uint8')
        if suffix == '.bin':
            if 'velodyne_points' in str(path):
                return inspect_lidar_bin(path)
            return inspect_raw_binary(path, np.uint8, 'bin_uint8')
    except Exception as exc:
        return {'error': str(exc)}
    return {'error': f'unsupported suffix {suffix}'}


def inspect_raw_binary(path, dtype, type_name):
    arr = np.fromfile(path, dtype=dtype)
    unique = np.unique(arr[: min(arr.size, 2_000_000)])
    return {
        'type': type_name,
        'num_values': int(arr.size),
        'dtype': str(arr.dtype),
        'min': safe_scalar(arr.min()) if arr.size else None,
        'max': safe_scalar(arr.max()) if arr.size else None,
        'unique_sample': [safe_scalar(x) for x in unique[:80]],
        'file_size_bytes': path.stat().st_size,
    }


def inspect_lidar_bin(path):
    arr = np.fromfile(path, dtype=np.float32)
    result = {
        'type': 'lidar_float32_bin',
        'num_float32': int(arr.size),
        'file_size_bytes': path.stat().st_size,
    }
    if arr.size % 4 == 0:
        points = arr.reshape(-1, 4)
        result['possible_shape_xyzi'] = list(points.shape)
        result['xyz_min'] = [safe_scalar(x) for x in points[:, :3].min(axis=0)]
        result['xyz_max'] = [safe_scalar(x) for x in points[:, :3].max(axis=0)]
    return result


def detect_presence(root, dirs, files):
    dir_names = {p.name.lower() for p in dirs}
    rel_names = [rel(p, root).lower() for p in files]
    ext_counter = Counter(p.suffix.lower() or '<no_ext>' for p in files)

    has_image = any(p.suffix.lower() in IMAGE_EXTS for p in files)
    has_lidar = any(p.suffix.lower() in LIDAR_EXTS for p in files) or any(
        token in dir_names for token in ('velodyne', 'lidar', 'pointcloud', 'points'))
    has_calib = any(
        ('calib' in name or 'camera' in name or 'intrinsic' in name)
        and Path(name).suffix.lower() in CALIB_EXTS
        for name in rel_names)
    has_split = any(token in dir_names for token in SPLIT_HINTS) or any(
        Path(name).stem.lower() in SPLIT_HINTS or any(f'/{token}' in name for token in SPLIT_HINTS)
        for name in rel_names)
    has_occ = any(
        p.suffix.lower() in ARRAY_EXTS.union(PKL_EXTS).union(RAW_LABEL_EXTS)
        and any(hint in rel(p, root).lower() for hint in OCC_HINTS)
        for p in files)

    return {
        'has_image': has_image,
        'has_lidar': has_lidar,
        'has_calibration': has_calib,
        'has_split': has_split,
        'has_occupancy_or_label': has_occ,
        'extension_counts': dict(sorted(ext_counter.items())),
        'directory_keywords': {
            key: key in dir_names for key in [
                'train', 'val', 'test', 'sequences', 'labels', 'voxels',
                'calib', 'poses', 'image', 'image_00', 'image_01',
                'velodyne', 'lidar'
            ]
        },
    }


def infer_case(presence):
    if (presence['has_image'] and presence['has_lidar']
            and presence['has_calibration'] and presence['has_occupancy_or_label']
            and presence['has_split']):
        return 'Case A'
    if (presence['has_image'] and presence['has_occupancy_or_label']
            and not presence['has_lidar']):
        return 'Case B'
    if presence['has_occupancy_or_label']:
        return 'Case C'
    return 'Insufficient'


def collect_shape_hints(inspections):
    shapes = defaultdict(int)
    class_ids = Counter()
    ignore_candidates = Counter()
    for item in inspections:
        payload = item.get('inspection', {})
        arrays = []
        path_name = rel(item.get('path'), item.get('root', Path('.'))).lower()
        if payload.get('type') == 'npz':
            arrays.extend(payload.get('keys', {}).items())
        elif payload.get('type') == 'npy':
            arrays.append(('array', payload.get('array', {})))
        for key, stats in arrays:
            name = key.lower()
            if any(hint in name for hint in OCC_HINTS) or any(
                    hint in path_name for hint in OCC_HINTS):
                shape = tuple(stats.get('shape', []))
                if shape:
                    shapes[str(shape)] += 1
                min_v = stats.get('min')
                max_v = stats.get('max')
                if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)):
                    class_ids[(int(min_v), int(max_v))] += 1
                    if max_v in (255, 65535):
                        ignore_candidates[int(max_v)] += 1
    return shapes, class_ids, ignore_candidates


def render_markdown(root, dirs, files, tree_lines, presence, samples,
                    inspections, inferred_case):
    shapes, class_ids, ignore_candidates = collect_shape_hints(inspections)
    lines = []
    lines.append('# SSCBench-KITTI Data Inspection')
    lines.append('')
    lines.append(f'- Data root: `{root}`')
    lines.append(f'- Total directories: {len(dirs)}')
    lines.append(f'- Total files: {len(files)}')
    lines.append(f'- Inferred case: **{inferred_case}**')
    lines.append('')
    lines.append('## Presence Summary')
    for key in ['has_image', 'has_lidar', 'has_calibration',
                'has_split', 'has_occupancy_or_label']:
        lines.append(f'- {key}: `{presence[key]}`')
    lines.append('')
    lines.append('## Extension Counts')
    for ext, count in presence['extension_counts'].items():
        lines.append(f'- `{ext}`: {count}')
    lines.append('')
    lines.append('## Directory Keyword Check')
    for key, value in presence['directory_keywords'].items():
        lines.append(f'- `{key}`: `{value}`')
    lines.append('')
    lines.append('## Directory Tree Summary')
    lines.append('```text')
    lines.extend(tree_lines)
    lines.append('```')
    lines.append('')
    lines.append('## Sample Files')
    for group, paths in samples.items():
        lines.append(f'### {group}')
        if not paths:
            lines.append('- None found')
        for path in paths:
            lines.append(f'- `{rel(path, root)}`')
        lines.append('')
    lines.append('## Array / PKL Inspection')
    for item in inspections:
        lines.append(f"### `{rel(item['path'], root)}`")
        lines.append('```json')
        lines.append(json.dumps(item['inspection'], indent=2, ensure_ascii=False))
        lines.append('```')
        lines.append('')
    lines.append('## Occupancy Shape Hints')
    if shapes:
        for shape, count in shapes.items():
            lines.append(f'- shape `{shape}` seen {count} time(s)')
    else:
        lines.append('- No reliable occupancy shape inferred from sampled arrays.')
    lines.append('')
    lines.append('## Class / Ignore Hints')
    if class_ids:
        for (min_v, max_v), count in class_ids.items():
            lines.append(f'- sampled label min/max `{min_v}` / `{max_v}` seen {count} time(s)')
    else:
        lines.append('- No class id range inferred from sampled arrays.')
    if ignore_candidates:
        for ignore, count in ignore_candidates.items():
            lines.append(f'- possible ignore index `{ignore}` seen {count} time(s)')
    else:
        lines.append('- No 255/65535 ignore index observed in sampled occupancy arrays.')
    lines.append('')
    lines.append('## Machine Readable Summary')
    lines.append('```json')
    lines.append(json.dumps({
        'data_root': str(root),
        'num_dirs': len(dirs),
        'num_files': len(files),
        'presence': presence,
        'inferred_case': inferred_case,
        'occupancy_shape_hints': dict(shapes),
        'class_id_hints': {f'{k[0]}..{k[1]}': v for k, v in class_ids.items()},
        'ignore_candidates': dict(ignore_candidates),
    }, indent=2, ensure_ascii=False))
    lines.append('```')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/sscbench-kitti')
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--sample-limit', type=int, default=8)
    parser.add_argument('--report-name', default='SSCBENCH_KITTI_INSPECTION.md')
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    dirs, files = walk_dataset(root)
    presence = detect_presence(root, dirs, files)
    tree_lines = limited_tree(dirs, files, root, max_depth=args.max_depth)

    samples = {
        'pkl': sample_files(files, lambda p: p.suffix.lower() in PKL_EXTS, args.sample_limit),
        'npz': sample_files(files, lambda p: p.suffix.lower() == '.npz', args.sample_limit),
        'npy': sample_files(files, lambda p: p.suffix.lower() == '.npy', args.sample_limit),
        'label': sample_files(files, lambda p: p.suffix.lower() == '.label', args.sample_limit),
        'invalid': sample_files(files, lambda p: p.suffix.lower() == '.invalid', args.sample_limit),
        'image': sample_files(files, lambda p: p.suffix.lower() in IMAGE_EXTS, args.sample_limit),
        'lidar': sample_files(files, lambda p: p.suffix.lower() in LIDAR_EXTS, args.sample_limit),
        'calibration_or_pose': sample_files(
            files,
            lambda p: p.suffix.lower() in CALIB_EXTS
            and any(token in rel(p, root).lower()
                    for token in ('calib', 'pose', 'camera', 'intrinsic')),
            args.sample_limit),
        'split': sample_files(
            files,
            lambda p: any(token in rel(p, root).lower().split('/')
                          or Path(p).stem.lower() == token for token in SPLIT_HINTS),
            args.sample_limit),
    }

    inspect_candidates = []
    for group in ('npz', 'npy', 'pkl', 'label', 'invalid', 'lidar'):
        inspect_candidates.extend(samples[group])
    seen = set()
    inspections = []
    for path in inspect_candidates:
        if path in seen:
            continue
        seen.add(path)
        inspections.append({
            'path': path,
            'root': root,
            'inspection': inspect_file(path)
        })

    inferred_case = infer_case(presence)
    report = render_markdown(
        root, dirs, files, tree_lines, presence, samples, inspections,
        inferred_case)
    report_path = root / args.report_name
    report_path.write_text(report, encoding='utf-8')

    print(report)
    print(f'\nSaved inspection report to: {report_path}')


if __name__ == '__main__':
    main()
