_base_ = ['./oup_occ_fusion_r18_tuned_mIoU_52_13.py']

find_unused_parameters = True

model = dict(
    type='OUPFusionOCC',
    use_uncertainty_refinement=True,
    img_backbone=dict(with_cp=False),
    pts_backbone=dict(with_cp=False),
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
        lambda_u=1.0),
    use_gaussian_residual=True,
    gaussian_residual=dict(
        type='UncertaintyGuidedGaussianResidual',
        in_channels=512,
        hidden_channels=128,
        Dz=16,
        num_classes=18,
        num_gaussians=512,
        topk_ratio=None,
        gaussian_sigma=2.0,
        aggregation='max',
        beta_gaussian=0.2,
        residual_clip=2.0,
        detach_uncertainty=True,
        use_high_uncertainty_mask=True,
        lambda_gaussian=1.0))

# Fine-tune from the tuned OUP-Occ 52.13 checkpoint.
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, start=1)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

load_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_gaussian_residual_100%_seqs'
