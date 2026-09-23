#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config
from PIL import Image

from mmdet3d.datasets.sscbench_kitti_dataset import load_sscbench_label


PALETTE = np.array([
    [0, 0, 0],
    [245, 150, 100],
    [245, 230, 100],
    [150, 60, 30],
    [180, 30, 80],
    [255, 0, 0],
    [30, 30, 255],
    [200, 40, 255],
    [90, 30, 150],
    [255, 0, 255],
    [255, 150, 255],
    [75, 0, 75],
    [75, 0, 175],
    [0, 200, 255],
    [50, 120, 255],
    [0, 175, 0],
    [0, 60, 135],
    [80, 240, 150],
    [150, 240, 255],
    [0, 0, 255],
    [255, 255, 255],
], dtype=np.uint8)


def bev_projection(volume, ignore_index=255):
    volume = np.asarray(volume)
    valid = volume != ignore_index
    z_index = np.arange(volume.shape[2], dtype=np.int32).reshape(1, 1, -1)
    z_index = np.where(valid, z_index, -1)
    top = z_index.argmax(axis=2)
    bev = np.take_along_axis(volume, top[..., None], axis=2)[..., 0]
    bev[top < 0] = ignore_index
    return bev


def colorize(bev, ignore_index=255):
    color_index = bev.copy()
    color_index[color_index == ignore_index] = len(PALETTE) - 1
    color_index = np.clip(color_index, 0, len(PALETTE) - 1)
    return PALETTE[color_index][::-1]


def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def load_prediction(pred_dir, sample_id):
    pred_dir = Path(pred_dir)
    safe_name = sample_id.replace('/', '_')
    for path in [
            pred_dir / f'{safe_name}.npy',
            pred_dir / f'{safe_name}.npz',
            pred_dir / f'{Path(sample_id).name}.npy',
            pred_dir / f'{Path(sample_id).name}.npz',
    ]:
        if not path.exists():
            continue
        if path.suffix == '.npy':
            return np.load(path), {}
        data = np.load(path)
        pred = data['pred'] if 'pred' in data else data[data.files[0]]
        aux = {key: data[key] for key in data.files if key != 'pred'}
        return pred, aux
    return None, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--pred-dir', default=None)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data.get(args.split, cfg.data.test)
    data_root = Path(data_cfg.data_root)
    infos_obj = mmcv.load(data_cfg.ann_file, file_format='pkl')
    infos = infos_obj['infos'] if isinstance(infos_obj, dict) else infos_obj
    if not infos:
        raise RuntimeError('No samples found. Run the converter after adding calibration and split files.')

    rng = random.Random(args.seed)
    selected = infos if len(infos) <= args.num_samples else rng.sample(infos, args.num_samples)
    out_dir = Path(args.out_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(args.checkpoint).parent / 'predictions'

    for info in selected:
        sample_id = info['sample_id']
        sample_dir = out_dir / sample_id.replace('/', '_')
        label, valid_mask = load_sscbench_label(
            str(data_root / info['occupancy_label_path']),
            invalid_path=str(data_root / info['invalid_path']) if info.get('invalid_path') else None,
            label_shape=info.get('voxel_shape', data_cfg.get('voxel_shape', (128, 128, 16))),
            label_dtype=info.get('label_dtype', 'uint16'),
            ignore_index=data_cfg.get('ignore_index', cfg.get('ignore_index', 255)))
        gt_bev = bev_projection(label)
        save_png(colorize(gt_bev), sample_dir / 'gt_bev.png')

        pred, aux = load_prediction(pred_dir, sample_id)
        if pred is not None:
            pred_bev = bev_projection(pred)
            save_png(colorize(pred_bev), sample_dir / 'pred_bev.png')
            error = np.zeros((*gt_bev.shape, 3), dtype=np.uint8)
            valid = np.logical_and(valid_mask.max(axis=2), gt_bev != 255)
            error[valid & (gt_bev == pred_bev)] = [0, 180, 0]
            error[valid & (gt_bev != pred_bev)] = [255, 0, 0]
            save_png(error[::-1], sample_dir / 'error_bev.png')
        for key, value in aux.items():
            if value.ndim >= 2:
                arr = np.asarray(value)
                if arr.ndim == 3:
                    arr = arr.max(axis=-1)
                arr = arr.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                save_png((arr * 255).astype(np.uint8), sample_dir / f'{key}.png')

    print(f'Saved visualization files to {out_dir}')
    if not pred_dir.exists():
        print(f'Prediction directory not found: {pred_dir}; saved GT only.')


if __name__ == '__main__':
    main()
