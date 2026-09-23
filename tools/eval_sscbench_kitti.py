#!/usr/bin/env python
import argparse
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config

from mmdet3d.datasets.sscbench_kitti_dataset import (
    SSCBenchKittiMetric, load_sscbench_label)


def load_predictions(pred_path):
    pred_path = Path(pred_path)
    if pred_path.is_file():
        data = mmcv.load(str(pred_path))
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data
    return pred_path


def load_prediction_from_dir(pred_dir, sample_id):
    safe_name = sample_id.replace('/', '_')
    candidates = [
        pred_dir / f'{safe_name}.npy',
        pred_dir / f'{safe_name}.npz',
        pred_dir / f'{Path(sample_id).name}.npy',
        pred_dir / f'{Path(sample_id).name}.npz',
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == '.npy':
            return np.load(path)
        data = np.load(path)
        for key in ('pred', 'prediction', 'occ_pred', 'arr_0'):
            if key in data:
                return data[key]
    raise FileNotFoundError(f'No prediction found for {sample_id} in {pred_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('predictions')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data.get(args.split, cfg.data.test)
    data_root = Path(data_cfg.data_root)
    ann_file = Path(args.ann_file or data_cfg.ann_file)
    infos_obj = mmcv.load(str(ann_file), file_format='pkl')
    infos = infos_obj['infos'] if isinstance(infos_obj, dict) else infos_obj

    class_names = data_cfg.get('classes', cfg.get('class_names', None))
    ignore_index = data_cfg.get('ignore_index', cfg.get('ignore_index', 255))
    metric = SSCBenchKittiMetric(class_names=class_names, ignore_index=ignore_index)
    predictions = load_predictions(args.predictions)

    for index, info in enumerate(infos):
        if isinstance(predictions, Path):
            pred = load_prediction_from_dir(predictions, info['sample_id'])
        else:
            pred = predictions[index]
        label_path = data_root / info['occupancy_label_path']
        invalid_path = data_root / info['invalid_path'] if info.get('invalid_path') else None
        label, valid_mask = load_sscbench_label(
            str(label_path),
            invalid_path=str(invalid_path) if invalid_path else None,
            label_shape=info.get('voxel_shape', data_cfg.get('voxel_shape', (128, 128, 16))),
            label_dtype=info.get('label_dtype', 'uint16'),
            ignore_index=ignore_index)
        metric.add_batch(pred, label, valid_mask=valid_mask)

    result = metric.results()
    out_dir = Path(args.out_dir or cfg.get('work_dir', './work_dirs/sscbench_kitti_eval'))
    out_dir.mkdir(parents=True, exist_ok=True)
    mmcv.dump(result, str(out_dir / 'eval_results.json'))
    metric.dump_csv(str(out_dir / 'per_class_iou.csv'))
    print(result)
    print(f'Saved eval_results.json and per_class_iou.csv to {out_dir}')


if __name__ == '__main__':
    main()
