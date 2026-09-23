"""Occ3D leave-one-out ablation: remove confidence uncertainty."""

_base_ = ['./oup_occ_fusion_r18_complete_version_mIoU_54_14.py']

model = dict(
    uncertainty_refinement=dict(
        # Keep the original semantic:visibility ratio (0.45:0.20).
        alpha_sem=0.69230769,
        alpha_occ=0.30769231,
        alpha_conf=0.0))

# Validation runs at epochs 40-48; retain all corresponding checkpoints.
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

load_from = None
resume_from = None
work_dir = 'work_dirs/ablation_uncertainty_no_confidence_100%_seqs'
