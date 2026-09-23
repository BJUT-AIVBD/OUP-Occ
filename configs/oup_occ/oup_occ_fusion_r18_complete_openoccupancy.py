_base_ = ['../effocc_openoccupancy/effocc-fusion-r18.py']

find_unused_parameters = True

# OpenOccupancy uses a 512 x 512 x 40 grid with 17 classes (free class = 0).
open_occ_head_cfg = dict(
    type='BEVOCCHead2DOpenOccupancy',
    in_dim=512,
    out_dim=256,
    Dz=40,
    use_mask=False,
    num_classes=17,
    use_predicter=True,
    class_wise=False,
    empty_idx=0)

coarse_open_occ_head_cfg = dict(
    type='BEVOCCHead2DOpenOccupancy',
    in_dim=512,
    out_dim=256,
    Dz=20,
    use_mask=False,
    num_classes=17,
    use_predicter=True,
    class_wise=False,
    empty_idx=0,
    # Coarse logits only provide auxiliary supervision and uncertainty.
    # Keeping CE alone avoids retaining three extra 3D softmax graphs.
    loss_weight_cfg=dict(
        loss_voxel_ce_weight=1.0,
        loss_voxel_sem_scal_weight=0.0,
        loss_voxel_geo_scal_weight=0.0,
        loss_voxel_lovasz_weight=0.0))

model = dict(
    type='OUPFusionOCC',
    mode='openoccupancy',
    use_camera_mask=False,
    use_lidar_mask=False,
    img_backbone=dict(with_cp=False),
    img_view_transformer=dict(with_cp=False),
    pts_backbone=dict(with_cp=False),
    use_uncertainty_refinement=True,
    coarse_occ_head=coarse_open_occ_head_cfg,
    final_occ_head=open_occ_head_cfg,
    # 256x256x20 coarse logits reduce the auxiliary branch memory by 8x.
    coarse_downsample_factor=2,
    use_reliability_fusion=True,
    reliability_fusion=dict(
        type='UncertaintyGuidedReliabilityFusion',
        img_channels=64,
        pts_channels=256,
        fused_channels=256,
        hidden_channels=32,
        # OpenOccupancy does not provide Occ3D-compatible camera/lidar masks.
        use_visibility=False,
        use_discrepancy=True,
        reliability_type='softmax',
        reliability_residual_scale=0.3,
        detach_visibility=True,
        lambda_reliability_reg=0.0),
    use_deformable_cross_attention=True,
    deformable_cross_attention=dict(
        type='UncertaintyGuidedDeformableCrossAttention',
        img_channels=64,
        pts_channels=256,
        in_channels=512,
        hidden_dim=32,
        num_queries=1024,
        topk_ratio=None,
        num_points=4,
        max_offset=4.0,
        use_reliability_prior=True,
        udca_residual_scale=0.3,
        # Dense 512-channel smoothing at 512 x 512 is prohibitively expensive.
        smooth_residual=False,
        detach_uncertainty=True,
        lambda_udca_sparse_reg=0.0),
    uncertainty_refinement=dict(
        type='UncertaintyGuidedBEVRefinement',
        in_channels=512,
        # Preserve the 0.45:0.35 semantic/confidence ratio after removing
        # Occ3D-specific visibility uncertainty.
        alpha_sem=0.5625,
        alpha_occ=0.0,
        alpha_conf=0.4375,
        projection='max',
        detach_uncertainty=True,
        gate_channels=1,
        gate_scale=1.0,
        num_refine_stages=2,
        lambda_coarse=0.3,
        # The official final head already contains four supervision terms.
        lambda_focus=0.0,
        lambda_u=1.0))

# Keep the official OpenOccupancy Fusion-R18 optimizer and 24-epoch schedule.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

fp16 = dict(loss_scale='dynamic')

load_from = None
resume_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_complete_openoccupancy'
