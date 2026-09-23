_base_ = ['../oup_occ_fusion_r18_tuned_mIoU_52_13.py']

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
        # Keep a small part of the tuned uncertainty focus while reducing UHAS strength.
        lambda_focus=0.02,
        lambda_u=1.0,
        use_hard_region_supervision=True,
        lambda_hard=0.05,
        hard_region_loss=dict(
            type='UncertaintyAwareHardRegionLoss',
            lambda_u=0.8,
            lambda_partial=0.2,
            lambda_invisible=0.0,
            lambda_far=0.2,
            lambda_class=0.2,
            lambda_boundary=0.0,
            use_boundary_weight=False,
            weight_max=2.5,
            normalize_weight=True,
            detach_uncertainty=True,
            ignore_index=255,
            point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])))

# Conservative fine-tuning from the tuned 52.13 checkpoint.
optimizer = dict(type='AdamW', lr=8e-5, weight_decay=1e-2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[6])
runner = dict(type='EpochBasedRunner', max_epochs=8)
evaluation = dict(interval=1, start=1)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1)

load_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_uhas_conservative_100%_seqs'
