#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


KITTI_CLASS_NAMES = [
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'terrain', 'pole', 'traffic-sign',
    'other-structure', 'other-object'
]

OFFICIAL_SPLITS = {
    'train': [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync',
        '2013_05_28_drive_0004_sync',
        '2013_05_28_drive_0005_sync',
        '2013_05_28_drive_0007_sync',
        '2013_05_28_drive_0010_sync',
    ],
    'val': ['2013_05_28_drive_0006_sync'],
    'test': ['2013_05_28_drive_0009_sync'],
}

OFFICIAL_P2 = np.array([
    552.554261, 0.0, 682.049453, 0.0,
    0.0, 552.554261, 238.769549, 0.0,
    0.0, 0.0, 1.0, 0.0,
], dtype=np.float32).reshape(3, 4)

OFFICIAL_CAM2VELO = np.array([
    0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
    -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
    -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824,
], dtype=np.float32).reshape(3, 4)


def rel(path, root):
    return str(Path(path).relative_to(root))


def collect_sequences(data_root):
    data_2d = data_root / 'data_2d_raw'
    if not data_2d.exists():
        return []
    return sorted([p.name for p in data_2d.iterdir() if p.is_dir()])


def collect_frames(data_root, sequence, label_scale, label_subdir):
    label_dir = data_root / label_subdir / sequence
    suffix = f'_{label_scale}.npy'
    if label_dir.exists():
        return sorted(p.name[:-len(suffix)] for p in label_dir.glob(f'*{suffix}'))
    voxel_dir = data_root / 'data_2d_raw' / sequence / 'voxels'
    if voxel_dir.exists():
        return sorted(p.stem for p in voxel_dir.glob('*.label'))
    return []


def find_image_paths(data_root, sequence, frame_id):
    paths = {}
    for cam_name, folder in [('CAM_LEFT', 'image_00')]:
        path = (data_root / 'data_2d_raw' / sequence / folder / 'data_rect' /
                f'{frame_id}.png')
        if path.exists():
            paths[cam_name] = rel(path, data_root)
    return paths


def find_lidar_path(data_root, sequence, frame_id):
    candidates = [
        data_root / 'data_3d_raw' / sequence / 'velodyne_points' / 'data' /
        f'{frame_id}.bin',
        data_root / 'data_2d_raw' / sequence / 'voxels' / f'{frame_id}.bin',
    ]
    for path in candidates:
        if path.exists():
            return rel(path, data_root)
    return None


def find_label_paths(data_root, sequence, frame_id, label_scale, label_subdir):
    npy = data_root / label_subdir / sequence / f'{frame_id}_{label_scale}.npy'
    raw_label = data_root / 'data_2d_raw' / sequence / 'voxels' / f'{frame_id}.label'
    invalid = data_root / 'data_2d_raw' / sequence / 'voxels' / f'{frame_id}.invalid'
    label = npy if npy.exists() else raw_label
    return (
        rel(label, data_root) if label.exists() else None,
        rel(invalid, data_root) if invalid.exists() else None,
    )


def find_split_files(data_root, split_dir=None):
    split_roots = []
    if split_dir is not None:
        split_roots.append(Path(split_dir).resolve())
    split_roots.extend([data_root / 'splits', data_root / 'ImageSets'])
    result = {}
    for split_root in split_roots:
        if not split_root.exists():
            continue
        for split in ('train', 'val', 'test'):
            path = split_root / f'{split}.txt'
            if path.exists():
                result[split] = path
    return result


def parse_split_file(path):
    items = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) == 1 and '/' in parts[0]:
            seq, frame = parts[0].split('/')[-2:]
        elif len(parts) >= 2:
            seq, frame = parts[0], parts[1]
        else:
            continue
        frame = Path(frame).stem
        items.append((seq, frame))
    return items


def find_calibration_files(data_root, calib_root=None):
    roots = []
    if calib_root is not None:
        roots.append(Path(calib_root).resolve())
    roots.extend([data_root, data_root / 'calibration'])
    names = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in {
                    '.txt', '.json', '.yaml', '.yml'
            }:
                continue
            lower = str(path).lower()
            if any(token in lower for token in ('calib', 'intrinsic', 'extrinsic', 'camera')):
                names.append(str(path))
    return names


def read_calib_matrix_file(path):
    matrices = {}
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, values = line.split(':', 1)
        else:
            parts = line.split()
            key, values = parts[0], ' '.join(parts[1:])
        nums = [float(x) for x in values.replace(',', ' ').split()]
        if len(nums) == 12:
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :4] = np.asarray(nums, dtype=np.float32).reshape(3, 4)
            matrices[key.strip()] = mat
        elif len(nums) == 9:
            matrices[key.strip()] = np.asarray(nums, dtype=np.float32).reshape(3, 3)
        elif len(nums) >= 2:
            matrices[key.strip()] = np.asarray(nums, dtype=np.float32)
    return matrices


def load_kitti360_calibration(calib_root):
    calib_root = Path(calib_root).resolve()
    perspective = calib_root / 'perspective.txt'
    cam_to_velo = calib_root / 'calib_cam_to_velo.txt'
    cam_to_pose = calib_root / 'calib_cam_to_pose.txt'

    if not perspective.exists() or not cam_to_velo.exists():
        return None

    perspective_data = read_calib_matrix_file(perspective)
    cam_to_velo_data = read_calib_matrix_file(cam_to_velo)
    pose_data = read_calib_matrix_file(cam_to_pose) if cam_to_pose.exists() else {}

    cam0_to_velo = None
    for value in cam_to_velo_data.values():
        if np.asarray(value).shape == (4, 4):
            cam0_to_velo = value
            break
    if cam0_to_velo is None:
        return None

    cam0_to_pose = pose_data.get('image_00', np.eye(4, dtype=np.float32))
    cams = {}
    for cam_name, kitti_name, p_key in [
            ('CAM_LEFT', 'image_00', 'P_rect_00'),
            ('CAM_RIGHT', 'image_01', 'P_rect_01')]:
        intrinsic = perspective_data.get(p_key)
        if intrinsic is None:
            intrinsic = np.eye(3, dtype=np.float32)
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]
        cam_to_pose_i = pose_data.get(kitti_name, cam0_to_pose)
        cam_i_to_cam0 = np.linalg.inv(cam0_to_pose) @ cam_to_pose_i
        sensor2lidar = cam0_to_velo @ cam_i_to_cam0
        cams[cam_name] = dict(
            sensor2lidar=sensor2lidar.astype(float).tolist(),
            cam_intrinsic=intrinsic[:3, :3].astype(float).tolist())
    return cams


def official_monoscene_calibration():
    cam2velo = np.concatenate(
        [OFFICIAL_CAM2VELO, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
        axis=0)
    return {
        'CAM_LEFT': dict(
            sensor2lidar=cam2velo.astype(float).tolist(),
            cam_intrinsic=OFFICIAL_P2[:3, :3].astype(float).tolist(),
            projection=OFFICIAL_P2.astype(float).tolist(),
            calibration_source='SSCBench MonoScene hard-coded KITTI-360 calibration')
    }


def default_calibration(image_paths):
    cams = {}
    for cam_name in image_paths:
        cams[cam_name] = dict(
            sensor2lidar=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            cam_intrinsic=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
    return cams


def build_info(data_root, sequence, frame_id, split, label_scale, label_subdir,
               allow_missing_lidar, allow_missing_calib, calibration):
    image_paths = find_image_paths(data_root, sequence, frame_id)
    lidar_path = find_lidar_path(data_root, sequence, frame_id)
    label_path, invalid_path = find_label_paths(
        data_root, sequence, frame_id, label_scale, label_subdir)
    if not image_paths or label_path is None:
        return None, 'missing image or occupancy label'
    if lidar_path is None and not allow_missing_lidar:
        return None, 'missing lidar'
    if calibration is None and not allow_missing_calib:
        return None, 'missing calibration'
    cams = calibration or default_calibration(image_paths)
    sample_id = f'{sequence}/{frame_id}'
    return dict(
        sample_id=sample_id,
        sequence_id=sequence,
        frame_id=frame_id,
        split=split,
        image_paths=image_paths,
        lidar_path=lidar_path,
        occupancy_label_path=label_path,
        invalid_path=invalid_path,
        cams={cam: cams[cam] for cam in image_paths if cam in cams},
        voxel_shape=[128, 128, 16],
        voxel_size=[0.4, 0.4, 0.4],
        point_cloud_range=[0.0, -25.6, -2.0, 51.2, 25.6, 4.4],
        class_names=KITTI_CLASS_NAMES,
        num_classes=len(KITTI_CLASS_NAMES),
        ignore_index=255,
        label_dtype='uint16',
        calibration_source=next(iter(cams.values())).get(
            'calibration_source', 'external_or_placeholder'),
    ), None


def dump_infos(path, infos, metadata, overwrite=False):
    if path.exists() and not overwrite:
        return False
    with open(path, 'wb') as f:
        pickle.dump(dict(infos=infos, metadata=metadata), f)
    return True


def write_report(path, payload):
    lines = [
        '# SSCBench-KITTI Conversion Report',
        '',
        f"- Data root: `{payload['data_root']}`",
        f"- Can convert: `{payload['can_convert']}`",
        f"- Reason: {payload['reason']}",
        '',
        '## Discovered Inputs',
        f"- Sequences: {payload['sequences']}",
        f"- Candidate frames with labels: {payload['candidate_frames']}",
        f"- Label subdir: `{payload['label_subdir']}`",
        f"- Split mode: `{payload['split_mode']}`",
        f"- Calibration source: {payload['calibration_source']}",
        f"- Calibration files: {payload['calibration_files'] or 'None'}",
        f"- Split files: {payload['split_files'] or 'None'}",
        '',
        '## Output Infos',
    ]
    for split, info in payload['outputs'].items():
        lines.append(f"- {split}: {info}")
    lines.extend([
        '',
        '## Machine Readable Summary',
        '```json',
        json.dumps(payload, indent=2, ensure_ascii=False),
        '```',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/sscbench-kitti')
    parser.add_argument('--out-dir', default='data/sscbench-kitti')
    parser.add_argument('--label-scale', default='1_2')
    parser.add_argument(
        '--label-subdir',
        default='preprocess/unified/labels',
        help='Prefer SSCBench unified labels by default.')
    parser.add_argument('--calib-root', default=None)
    parser.add_argument('--split-dir', default=None)
    parser.add_argument(
        '--split-mode',
        default='official',
        choices=['official', 'files', 'sequence'],
        help='official uses SSCBench train 00/02/03/04/05/07/10, val 06, test 09.')
    parser.add_argument(
        '--calib-mode',
        default='official',
        choices=['official', 'files', 'placeholder'],
        help='official uses the KITTI-360 constants from SSCBench MonoScene.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-missing-lidar', action='store_true')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = collect_sequences(data_root)
    candidate_frames = {
        seq: len(collect_frames(data_root, seq, args.label_scale, args.label_subdir))
        for seq in sequences
    }
    split_files = find_split_files(data_root, split_dir=args.split_dir)
    calibration_files = find_calibration_files(data_root, calib_root=args.calib_root)
    if args.calib_mode == 'official':
        calibration = official_monoscene_calibration()
        calibration_source = 'SSCBench MonoScene hard-coded KITTI-360 calibration'
    elif args.calib_mode == 'files':
        calibration = load_kitti360_calibration(args.calib_root) if args.calib_root else None
        if calibration is None and (data_root / 'calibration').exists():
            calibration = load_kitti360_calibration(data_root / 'calibration')
        calibration_source = 'external KITTI-360 calibration files'
    else:
        calibration = None
        calibration_source = 'placeholder identity calibration'
    outputs = {}

    can_convert = True
    blockers = []
    if args.split_mode == 'files' and not split_files:
        can_convert = False
        blockers.append('missing train/val/test split files')
    if args.calib_mode == 'files' and calibration is None:
        can_convert = False
        blockers.append('missing usable KITTI-360 calibration files')

    if can_convert:
        if args.split_mode == 'files':
            split_items = {
                split: parse_split_file(path)
                for split, path in split_files.items()
            }
        elif args.split_mode == 'official':
            split_items = {
                split: [(seq, frame)
                        for seq in split_sequences
                        for frame in collect_frames(
                            data_root, seq, args.label_scale, args.label_subdir)]
                for split, split_sequences in OFFICIAL_SPLITS.items()
            }
        else:
            train_sequences = sequences[:-1]
            val_sequences = sequences[-1:]
            split_items = {
                'train': [(seq, frame) for seq in train_sequences
                          for frame in collect_frames(
                              data_root, seq, args.label_scale, args.label_subdir)],
                'val': [(seq, frame) for seq in val_sequences
                        for frame in collect_frames(
                            data_root, seq, args.label_scale, args.label_subdir)],
                'test': [],
            }

        metadata = dict(
            dataset='SSCBench-KITTI',
            version='kitti360',
            class_names=KITTI_CLASS_NAMES,
            label_scale=args.label_scale,
            label_subdir=args.label_subdir,
            split_mode=args.split_mode,
            calibration_source=calibration_source,
        )
        for split in ('train', 'val', 'test'):
            infos = []
            skipped = []
            for sequence, frame_id in split_items.get(split, []):
                info, reason = build_info(
                    data_root,
                    sequence,
                    frame_id,
                    split,
                    args.label_scale,
                    args.label_subdir,
                    allow_missing_lidar=args.allow_missing_lidar,
                    allow_missing_calib=args.calib_mode == 'placeholder',
                    calibration=calibration)
                if info is None:
                    skipped.append(dict(sequence=sequence, frame_id=frame_id, reason=reason))
                else:
                    infos.append(info)
            out_path = out_dir / f'sscbench_kitti_infos_{split}.pkl'
            written = dump_infos(out_path, infos, metadata, overwrite=args.overwrite)
            outputs[split] = dict(path=str(out_path), samples=len(infos),
                                  skipped=len(skipped), written=written)
    else:
        for split in ('train', 'val', 'test'):
            outputs[split] = 'not generated'

    payload = dict(
        data_root=str(data_root),
        can_convert=can_convert,
        reason='; '.join(blockers) if blockers else 'ok',
        sequences=sequences,
        candidate_frames=candidate_frames,
        label_subdir=args.label_subdir,
        split_mode=args.split_mode,
        calibration_source=calibration_source,
        calibration_files=calibration_files,
        calibration_loaded=calibration is not None,
        split_files={k: str(v) for k, v in split_files.items()},
        outputs=outputs,
    )
    report_path = out_dir / 'SSCBENCH_KITTI_CONVERSION_REPORT.md'
    write_report(report_path, payload)
    print(f'Wrote conversion report to {report_path}')
    if not can_convert:
        print('Info pkl files were not generated because required inputs are missing.')


if __name__ == '__main__':
    main()
