#!/usr/bin/env python
import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


CAMERAS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate nuScenes multi-view LiDAR depth ground truth.')
    parser.add_argument(
        '--data-root',
        default='data/nuscenes',
        help='nuScenes root containing samples, sweeps and bevdet info files.')
    parser.add_argument(
        '--info-prefix',
        default='bevdetv3-nuscenes',
        help='Info prefix, e.g. bevdetv3-nuscenes.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val', 'test'],
        help='Splits to process.')
    parser.add_argument(
        '--out-dir',
        default='data/nuscenes/depth_gt',
        help='Output directory. Files are saved as <out-dir>/<split>/<token>.npz.')
    parser.add_argument(
        '--mode',
        default='bevdet-test',
        choices=['bevdet-test', 'raw'],
        help='bevdet-test saves deterministic resized/cropped maps matching the '
        'OpenOccupancy test pipeline; raw saves maps in original image size.')
    parser.add_argument(
        '--input-size',
        nargs=2,
        type=int,
        default=[256, 704],
        metavar=('HEIGHT', 'WIDTH'),
        help='Target image size for bevdet-test mode.')
    parser.add_argument(
        '--resize-test',
        type=float,
        default=0.0,
        help='Extra resize_test ratio used by BEVDet test augmentation.')
    parser.add_argument(
        '--crop-h',
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=('LOW', 'HIGH'),
        help='BEVDet crop_h config. Test mode uses the mean value.')
    parser.add_argument(
        '--depth-range',
        nargs=2,
        type=float,
        default=[1.0, 60.0],
        metavar=('MIN', 'MAX'),
        help='Valid depth range in meters.')
    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Downsample factor for saved depth maps.')
    parser.add_argument(
        '--dtype',
        default='float16',
        choices=['float16', 'float32'],
        help='Saved gt_depth dtype.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate files that already exist.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Debug option. Process at most this many samples per split.')
    return parser.parse_args()


def resolve_path(path, data_root):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(data_root, path)


def load_points(path):
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f'Point file {path} size is not divisible by 5.')
    return points.reshape(-1, 5)[:, :3]


def image_size(path):
    with Image.open(path) as img:
        width, height = img.size
    return height, width


def deterministic_bevdet_transform(src_h, src_w, input_size, resize_test, crop_h_cfg):
    target_h, target_w = input_size
    resize = float(target_w) / float(src_w) + float(resize_test)
    new_w, new_h = int(src_w * resize), int(src_h * resize)
    crop_h = int((1 - float(np.mean(crop_h_cfg))) * new_h) - target_h
    crop_w = int(max(0, new_w - target_w) / 2)
    return resize, crop_w, crop_h, target_h, target_w


def project_lidar_to_camera(points_lidar, cam_info):
    cam2lidar = np.eye(4, dtype=np.float32)
    cam2lidar[:3, :3] = np.asarray(
        cam_info['sensor2lidar_rotation'], dtype=np.float32)
    cam2lidar[:3, 3] = np.asarray(
        cam_info['sensor2lidar_translation'], dtype=np.float32)
    lidar2cam = np.linalg.inv(cam2lidar)

    points_cam = points_lidar @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
    depth = points_cam[:, 2]
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
    points_img = points_cam @ intrinsic.T
    uv = points_img[:, :2] / np.maximum(points_img[:, 2:3], 1e-6)
    return uv[:, 0], uv[:, 1], depth


def points_to_depthmap(u, v, depth, height, width, depth_range, downsample):
    out_h, out_w = height // downsample, width // downsample
    depth_map = np.zeros((out_h, out_w), dtype=np.float32)
    coor_x = np.round(u / downsample).astype(np.int64)
    coor_y = np.round(v / downsample).astype(np.int64)
    kept = (
        (coor_x >= 0)
        & (coor_x < out_w)
        & (coor_y >= 0)
        & (coor_y < out_h)
        & (depth >= depth_range[0])
        & (depth < depth_range[1]))
    if not np.any(kept):
        return depth_map

    coor_x = coor_x[kept]
    coor_y = coor_y[kept]
    depth = depth[kept]
    ranks = coor_x + coor_y * out_w
    order = np.argsort(ranks + depth / 100.0)
    coor_x = coor_x[order]
    coor_y = coor_y[order]
    depth = depth[order]
    ranks = ranks[order]
    keep_first = np.ones(ranks.shape[0], dtype=bool)
    keep_first[1:] = ranks[1:] != ranks[:-1]
    depth_map[coor_y[keep_first], coor_x[keep_first]] = depth[keep_first]
    return depth_map


def process_one(task):
    info, split, args_dict = task
    data_root = args_dict['data_root']
    out_dir = args_dict['out_dir']
    token = info['token']
    out_path = os.path.join(out_dir, split, f'{token}.npz')
    if os.path.exists(out_path) and not args_dict['overwrite']:
        return 'skip'

    lidar_path = resolve_path(info['lidar_path'], data_root)
    points = load_points(lidar_path)
    depth_maps = []
    valid_counts = []
    for cam_name in args_dict['cameras']:
        cam_info = info['cams'][cam_name]
        img_path = resolve_path(cam_info['data_path'], data_root)
        src_h, src_w = image_size(img_path)
        u, v, depth = project_lidar_to_camera(points, cam_info)

        if args_dict['mode'] == 'bevdet-test':
            resize, crop_w, crop_h, out_h, out_w = deterministic_bevdet_transform(
                src_h,
                src_w,
                args_dict['input_size'],
                args_dict['resize_test'],
                args_dict['crop_h'])
            u = u * resize - crop_w
            v = v * resize - crop_h
        else:
            out_h, out_w = src_h, src_w

        depth_map = points_to_depthmap(
            u,
            v,
            depth,
            out_h,
            out_w,
            args_dict['depth_range'],
            args_dict['downsample'])
        valid_counts.append(int((depth_map > 0).sum()))
        depth_maps.append(depth_map)

    gt_depth = np.stack(depth_maps, axis=0)
    if args_dict['dtype'] == 'float16':
        gt_depth = gt_depth.astype(np.float16)
    else:
        gt_depth = gt_depth.astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        gt_depth=gt_depth,
        cam_names=np.asarray(args_dict['cameras']),
        valid_counts=np.asarray(valid_counts, dtype=np.int32))
    return 'write'


def main():
    args = parse_args()
    args.data_root = os.path.normpath(args.data_root)
    args.out_dir = os.path.normpath(args.out_dir)
    args_dict = vars(args).copy()
    args_dict['cameras'] = CAMERAS
    args_dict['input_size'] = tuple(args.input_size)
    args_dict['crop_h'] = tuple(args.crop_h)
    args_dict['depth_range'] = tuple(args.depth_range)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    for split in args.splits:
        info_path = os.path.join(
            args.data_root, f'{args.info_prefix}_infos_{split}.pkl')
        if not os.path.exists(info_path):
            raise FileNotFoundError(info_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        infos = data['infos']
        if args.max_samples is not None and args.max_samples > 0:
            infos = infos[:args.max_samples]
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        tasks = [(info, split, args_dict) for info in infos]
        stats = {'skip': 0, 'write': 0}
        if args.num_workers <= 1:
            iterator = map(process_one, tasks)
            for status in tqdm(iterator, total=len(tasks), desc=split):
                stats[status] = stats.get(status, 0) + 1
        else:
            with Pool(args.num_workers) as pool:
                iterator = pool.imap_unordered(process_one, tasks)
                for status in tqdm(iterator, total=len(tasks), desc=split):
                    stats[status] = stats.get(status, 0) + 1
        print(f'{split}: wrote {stats.get("write", 0)}, skipped {stats.get("skip", 0)}')


if __name__ == '__main__':
    main()
