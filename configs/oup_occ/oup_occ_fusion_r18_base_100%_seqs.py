_base_ = ['../effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# SECOND uses activation checkpointing by default in this repo. We explicitly
# disable pts_backbone.with_cp below, so DDP unused-parameter detection can stay
# enabled for occasional sparse LiDAR branch rank differences.
find_unused_parameters = True

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
voxel_size = [0.05, 0.05, 0.16]
numC_Trans = 64
img_feat_dim = 128

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

file_client_args = dict(backend='disk')
bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

occ_head_cfg = dict(
    type='BEVOCCHead2D',
    in_dim=512,
    out_dim=256,
    Dz=16,
    use_mask=True,
    num_classes=18,
    use_predicter=True,
    class_wise=False,
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        ignore_index=255,
        loss_weight=1.0),
)

model = dict(
    _delete_=True,
    type='OUPFusionOCC',
    use_uncertainty_refinement=True,
    mode='occ_challenge',
    img_backbone=dict(
        pretrained='ckpts/torchvision/resnet18-f37072fd.pth',
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[128, 256, 512],
        out_channels=img_feat_dim,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_feat_dim,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=8,
        with_depth_from_lidar=True),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1600, 1600],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        with_cp=False),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    occ_fuser=dict(
        type='ConvFuser2D',
        in_channels=[64, 256],
        out_channels=256),
    coarse_occ_head=occ_head_cfg,
    final_occ_head=occ_head_cfg,
    uncertainty_refinement=dict(
        type='UncertaintyGuidedBEVRefinement',
        in_channels=512,
        alpha_sem=0.4,
        alpha_occ=0.3,
        alpha_conf=0.3,
        projection='max',
        detach_uncertainty=True,
        gate_channels=1,
        num_refine_stages=1,
        lambda_coarse=0.4,
        lambda_focus=0.0,
        lambda_u=1.0),
)

# The OUP model can use visibility masks during validation inference. The
# original baseline config remains unchanged.
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='PointToMultiViewDepthFusion',
        downsample=1,
        grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'points', 'img_inputs', 'gt_depth', 'voxel_semantics',
                    'mask_lidar', 'mask_camera'
                ])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

checkpoint_config = dict(interval=1, max_keep_ckpts=5)
load_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_base_100%_seqs'
