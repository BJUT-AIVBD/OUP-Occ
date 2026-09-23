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
        lambda_focus=0.0,
        lambda_u=1.0,
        use_hard_region_supervision=True,
        lambda_hard=0.08,
        hard_region_loss=dict(
            type='UncertaintyAwareHardRegionLoss',
            lambda_u=1.0,
            lambda_partial=0.3,
            lambda_invisible=0.0,
            lambda_far=0.3,
            lambda_class=0.3,
            lambda_boundary=0.0,
            use_boundary_weight=False,
            weight_max=3.0,
            normalize_weight=True,
            detach_uncertainty=True,
            ignore_index=255,
            point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])))

# Fine-tune from the tuned OUP-Occ checkpoint with a smaller learning rate.
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
work_dir = 'work_dirs/oup_occ_fusion_r18_uhas_100%_seqs'
