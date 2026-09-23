# OUP Occupancy Refinement

Recommended hardware stability settings for two-GPU training:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 220
sudo nvidia-smi -i 1 -pl 220
```

The OUP config sets `find_unused_parameters=True` and `pts_backbone.with_cp=False`.
This keeps DDP robust to occasional sparse LiDAR branch rank differences while
avoiding the `Expected to mark a variable ready only once` error caused by
combining activation checkpointing with DDP unused-parameter detection.

Train from scratch:

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_fusion_r18_base_100%_seqs.py \
  2
```

Resume:

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_fusion_r18_base_100%_seqs.py \
  2 \
  --resume-from work_dirs/oup_occ_fusion_r18_base_100%_seqs/latest.pth
```

Fine-tune from the official EFFOcc checkpoint:

```bash
bash tools/dist_train.sh \
  configs/oup_occ/oup_occ_fusion_r18_base_100%_seqs.py \
  2 \
  --cfg-options load_from=ckpts/effocc_fusion_r18.pth
```

Test:

```bash
bash tools/dist_test.sh \
  configs/oup_occ/oup_occ_fusion_r18_base_100%_seqs.py \
  work_dirs/oup_occ_fusion_r18_base_100%_seqs/latest.pth \
  2 \
  --eval mIoU
```

Analyze refinement effect:

```bash
python tools/analyze_refinement_effect.py \
  configs/oup_occ/oup_occ_fusion_r18_base_100%_seqs.py \
  work_dirs/oup_occ_fusion_r18_base_100%_seqs/latest.pth \
  --baseline-config configs/effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py \
  --baseline-ckpt ckpts/effocc_fusion_r18.pth \
  --out-dir work_dirs/oup_occ_refinement_analysis \
  --split val \
  --max-samples 200 \
  --vis-topk 30
```
