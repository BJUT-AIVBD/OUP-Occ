_base_ = ['./oup_occ_fusion_r18_complete_version_mIoU_54_14.py']

custom_imports = dict(
    imports=['mmdet3d.datasets.sscbench_kitti_dataset'],
    allow_failed_imports=False)

find_unused_parameters = True

dataset_type = 'SSCBenchKittiDataset'
data_root = 'data/sscbench-kitti/'

class_names = [
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'terrain', 'pole', 'traffic-sign',
    'other-structure', 'other-object'
]
num_classes = len(class_names)
ignore_index = 255

point_cloud_range = [0.0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.05, 0.05, 0.16]
ssc_voxel_shape = [128, 128, 16]
ssc_voxel_size = [0.4, 0.4, 0.4]

data_config = dict(
    cams=['CAM_LEFT'],
    Ncams=1,
    input_size=(256, 704),
    src_size=(376, 1408),
    resize=(-0.04, 0.08),
    rot=(-2.0, 2.0),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)

grid_config = dict(
    x=[0.0, 51.2, 0.4],
    y=[-25.6, 25.6, 0.4],
    z=[-2.0, 4.4, 6.4],
    depth=[1.0, 60.0, 0.5])

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

file_client_args = dict(backend='disk')
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)

occ_head_cfg = dict(
    type='BEVOCCHead2D',
    in_dim=512,
    out_dim=256,
    Dz=16,
    use_mask=True,
    num_classes=num_classes,
    use_predicter=True,
    class_wise=False,
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        ignore_index=ignore_index,
        loss_weight=1.0))

model = dict(
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(60000, 90000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        in_channels=4,
        sparse_shape=[41, 1024, 1024]),
    img_view_transformer=dict(
        grid_config=grid_config,
        input_size=data_config['input_size'],
        with_depth_from_lidar=True),
    coarse_occ_head=occ_head_cfg,
    final_occ_head=occ_head_cfg,
    use_reliability_fusion=True,
    reliability_fusion=dict(
        img_channels=64,
        pts_channels=256,
        fused_channels=256),
    use_deformable_cross_attention=True,
    deformable_cross_attention=dict(
        img_channels=64,
        pts_channels=256,
        in_channels=512,
        use_reliability_prior=True),
    use_uncertainty_refinement=True)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        use_lidar_coord=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadSSCBenchKittiOccGT',
        label_shape=ssc_voxel_shape,
        ignore_index=ignore_index),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='PointToMultiViewDepthFusion',
        use_lidar_coord=True,
        downsample=1,
        grid_config=grid_config),
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
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        use_lidar_coord=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadSSCBenchKittiOccGT',
        label_shape=ssc_voxel_shape,
        ignore_index=ignore_index),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='PointToMultiViewDepthFusion',
        use_lidar_coord=True,
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
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sscbench_kitti_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        voxel_shape=ssc_voxel_shape,
        voxel_size=ssc_voxel_size,
        point_cloud_range=point_cloud_range,
        ignore_index=ignore_index),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sscbench_kitti_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        voxel_shape=ssc_voxel_shape,
        voxel_size=ssc_voxel_size,
        point_cloud_range=point_cloud_range,
        ignore_index=ignore_index),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sscbench_kitti_infos_test.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        voxel_shape=ssc_voxel_shape,
        voxel_size=ssc_voxel_size,
        point_cloud_range=point_cloud_range,
        ignore_index=ignore_index))

optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=24)
total_epochs = 24

checkpoint_config = dict(interval=1, max_keep_ckpts=5)
load_from = None
resume_from = None
work_dir = 'work_dirs/oup_occ_complete_sscbench_kitti_100%_seqs'
