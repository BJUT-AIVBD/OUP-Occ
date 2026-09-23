# 相关文件可参考EFFOCC:https://github.com/synsin0/effocc
# SSCBench-KITTI / KITTI-360 OUP-Occ 适配说明

数据检查：

```bash
cd /home/mm/EFFOcc
python tools/inspect_sscbench_kitti.py \
  --data-root data/sscbench-kitti
```

数据转换：

```bash
python tools/create_data_sscbench_kitti.py \
  --data-root data/sscbench-kitti \
  --out-dir data/sscbench-kitti \
  --overwrite
```

该命令默认使用 SSCBench 官方 split 和 MonoScene hard-coded KITTI-360 calibration，生成：

```text
data/sscbench-kitti/sscbench_kitti_infos_train.pkl  # 8483 samples
data/sscbench-kitti/sscbench_kitti_infos_val.pkl    # 1812 samples
data/sscbench-kitti/sscbench_kitti_infos_test.pkl   # 2165 samples
```

如果你要改为外部 calibration 文件模式，需要的文件名按 KITTI-360 常见格式解析：

```text
data/sscbench-kitti/calibration/perspective.txt
data/sscbench-kitti/calibration/calib_cam_to_velo.txt
data/sscbench-kitti/calibration/calib_cam_to_pose.txt
```

需要的 split 文件：

```text
data/sscbench-kitti/splits/train.txt
data/sscbench-kitti/splits/val.txt
data/sscbench-kitti/splits/test.txt
```

如果你要改为外部 split 文件模式，需要：

```text
data/sscbench-kitti/splits/train.txt
data/sscbench-kitti/splits/val.txt
data/sscbench-kitti/splits/test.txt
```

权重转换：

```bash
python tools/convert_oup_occ_nuscenes_to_kitti_init.py \
  --src /home/mm/EFFOcc/work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/oup_occ_complete_version_mIoU_54_14.pth \
  --config /home/mm/EFFOcc/configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  --out /home/mm/EFFOcc/work_dirs/oup_occ_kitti_init_from_nuscenes_54_14.pth
```

训练前环境：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 220
sudo nvidia-smi -i 1 -pl 220
```

训练：

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  2 \
  --load-from work_dirs/oup_occ_kitti_init_from_nuscenes_54_14.pth
```

从头训练时去掉 `--load-from`。

中断恢复：

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  2 \
  --resume-from work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth
```

测试：

```bash
bash tools/dist_test.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth \
  2 \
  --eval mIoU \
  --eval-options out_dir=work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs
```

独立评估已有预测：

```bash
python tools/eval_sscbench_kitti.py \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/predictions \
  --out-dir work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs
```

可视化：

```bash
python tools/vis_sscbench_kitti_occ.py \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth \
  --out-dir work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/vis \
  --num-samples 20
```

完整运行顺序

参考资料：

- SSCBench 官方仓库：`https://github.com/ai4ce/SSCBench`
- SSCBench-KITTI-360 数据说明：`https://github.com/ai4ce/SSCBench/blob/main/dataset/KITTI-360/README.md`
- SSCBench KITTI-360 类别与 split 配置：`https://github.com/ai4ce/SSCBench/blob/main/dataset/configs/kitti360.yaml`
- MonoScene KITTI-360 dataset 参考实现：`https://github.com/ai4ce/SSCBench/blob/main/method/MonoScene/monoscene/data/kitti_360/kitti_360_dataset.py`

1. 进入仓库


2. 检查数据结构：

```bash
python tools/inspect_sscbench_kitti.py \
  --data-root data/sscbench-kitti
```

3. 生成 pkl：

```bash
python tools/create_data_sscbench_kitti.py \
  --data-root data/sscbench-kitti \
  --out-dir data/sscbench-kitti \
  --overwrite
```

4. 转换 nuScenes checkpoint 为 KITTI 初始化权重：

```bash
python tools/convert_oup_occ_nuscenes_to_kitti_init.py \
  --src /home/mm/EFFOcc/work_dirs/oup_occ_fusion_r18_ucrf_udca_100%_seqs/oup_occ_complete_version_mIoU_54_14.pth \
  --config /home/mm/EFFOcc/configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  --out /home/mm/EFFOcc/work_dirs/oup_occ_kitti_init_from_nuscenes_54_14.pth \
  --overwrite
```

5. 设置多卡训练环境：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 220
sudo nvidia-smi -i 1 -pl 220
```

6. 训练：

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  2 \
  --load-from work_dirs/oup_occ_kitti_init_from_nuscenes_54_14.pth
```

7. 如权重转换失败或你想从头训练：

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  2
```

8. 中断恢复：

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  2 \
  --resume-from work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth
```

9. 测试并调用 dataset evaluator：

```bash
bash tools/dist_test.sh \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth \
  2 \
  --eval mIoU \
  --eval-options out_dir=work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs
```

10. 独立评估已有预测目录：

```bash
python tools/eval_sscbench_kitti.py \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/predictions \
  --out-dir work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs
```

11. 可视化：

```bash
python tools/vis_sscbench_kitti_occ.py \
  configs/oup_occ/oup_occ_complete_sscbench_kitti_100%_seqs.py \
  work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/latest.pth \
  --out-dir work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs/vis \
  --num-samples 20
```
