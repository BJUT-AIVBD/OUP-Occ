_base_ = ['./oup_occ_fusion_r18_tuned_mIoU_52_13.py']

find_unused_parameters = True

model = dict(
    type='OUPFusionOCC',
    use_uncertainty_refinement=True,
    img_backbone=dict(with_cp=False),
    pts_backbone=dict(with_cp=False),
    use_reliability_fusion=True,
    reliability_fusion=dict(
        type='UncertaintyGuidedReliabilityFusion',
        img_channels=64,
        pts_channels=256,
        fused_channels=256,
        hidden_channels=64,
        use_visibility=True,
        use_discrepancy=True,
        use_uncertainty=False,
        reliability_type='softmax',
        reliability_residual_scale=0.3,
        detach_visibility=True,
        detach_discrepancy=False,
        lambda_reliability_reg=0.0),
    uncertainty_refinement=dict(
        type='UncertaintyGuidedBEVRefinement',
        in_channels=512,
        alpha_sem=0.45,
        alpha_occ=0.20,
        alpha_conf=0.35,
        projection='max',
        detach_uncertainty=True,
        gate_channels=1,
        gate_scale=1.0,
        num_refine_stages=2,
        lambda_coarse=0.3,
        lambda_focus=0.05,
        lambda_u=1.0))

# Full training from scratch with reliability-aware fusion enabled.
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=48)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)

load_from = None
resume_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_ucrf_100%_seqs'
