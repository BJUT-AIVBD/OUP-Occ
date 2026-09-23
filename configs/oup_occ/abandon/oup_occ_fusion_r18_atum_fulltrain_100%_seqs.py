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
    use_temporal_uncertainty_memory=True,
    temporal_memory=dict(
        type='AdaptiveTemporalUncertaintyMemory',
        gamma=0.8,
        memory_weight=0.4,
        use_adaptive_gate=True,
        detach_memory=True,
        reset_on_scene_change=True,
        max_memory_scenes=50,
        train_memory_mode='batch_local',
        test_memory_mode='scene_cache',
        use_pose_warp=False,
        lambda_memory_consistency=0.0))

# Full training from scratch with ATUM enabled.
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
work_dir = 'work_dirs/oup_occ_fusion_r18_atum_fulltrain_100%_seqs'
