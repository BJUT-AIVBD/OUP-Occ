"""Occ3D leave-one-out ablation: remove semantic uncertainty."""

_base_ = ['./oup_occ_fusion_r18_complete_version_mIoU_54_14.py']

model = dict(
    uncertainty_refinement=dict(
        # Keep the original visibility:confidence ratio (0.20:0.35).
        alpha_sem=0.0,
        alpha_occ=0.36363636,
        alpha_conf=0.63636364))

# Validation runs at epochs 40-48; retain all corresponding checkpoints.
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

load_from = None
resume_from = None
work_dir = 'work_dirs/ablation_uncertainty_no_semantic_100%_seqs'
