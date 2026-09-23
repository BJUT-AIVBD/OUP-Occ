"""Occ3D leave-one-out ablation: remove visibility uncertainty."""

_base_ = ['./oup_occ_fusion_r18_complete_version_mIoU_54_14.py']

model = dict(
    uncertainty_refinement=dict(
        # Keep the original semantic:confidence ratio (0.45:0.35).
        alpha_sem=0.5625,
        alpha_occ=0.0,
        alpha_conf=0.4375))

# Validation runs at epochs 40-48; retain all corresponding checkpoints.
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

load_from = None
resume_from = None
work_dir = 'work_dirs/ablation_uncertainty_no_visibility_100%_seqs'
