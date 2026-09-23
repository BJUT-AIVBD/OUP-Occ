_base_ = ['./oup_occ_fusion_r18_tuned_mIoU_52_13.py']

find_unused_parameters = True

model = dict(
    type='OUPFusionOCC',
    use_uncertainty_refinement=False,
    img_backbone=dict(with_cp=False),
    pts_backbone=dict(with_cp=False),
    use_reliability_fusion=False,
    use_deformable_cross_attention=True,
    deformable_cross_attention=dict(
        type='UncertaintyGuidedDeformableCrossAttention',
        img_channels=64,
        pts_channels=256,
        in_channels=512,
        hidden_dim=64,
        num_queries=1024,
        topk_ratio=None,
        num_points=4,
        max_offset=4.0,
        use_reliability_prior=False,
        udca_residual_scale=0.3,
        smooth_residual=True,
        detach_uncertainty=True,
        lambda_udca_sparse_reg=0.0),
    uncertainty_refinement=dict(
        type='UncertaintyGuidedBEVRefinement',
        in_channels=512,
        alpha_sem=0.45,
        alpha_occ=0.20,
        alpha_conf=0.35,
        projection='max',
        detach_uncertainty=True,
        gate_channels=1,
        gate_scale=0.0,
        num_refine_stages=1,
        lambda_coarse=0.3,
        lambda_focus=0.0,
        lambda_u=1.0,
        use_hard_region_supervision=False,
        lambda_hard=0.0))

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
work_dir = 'work_dirs/ablation_udca_only_100%_seqs'
