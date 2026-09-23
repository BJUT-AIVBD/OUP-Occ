_base_ = ['./oup_occ_fusion_r18_complete_version_mIoU_54_14.py']

model = dict(
    img_backbone=dict(
        pretrained='ckpts/torchvision/resnet50-0676ba61.pth',
        depth=50,
        with_cp=False),
    img_neck=dict(
        in_channels=[512, 1024, 2048],
        out_channels=256),
    img_view_transformer=dict(
        in_channels=256,
        with_depth_from_lidar=False,
        large_depth_net=True),
    pts_middle_encoder=dict(
        base_channels=32,
        output_channels=256,
        encoder_channels=((32, 32, 64), (64, 64, 128),
                          (128, 128, 256), (256, 256))),
    pts_backbone=dict(
        in_channels=512,
        out_channels=[128, 256, 256],
        layer_nums=[3, 3, 3],
        layer_strides=[1, 2, 2],
        with_cp=False),
    pts_neck=dict(
        in_channels=[128, 256, 256],
        out_channels=[128, 128, 256],
        upsample_strides=[1, 2, 4]),
    occ_fuser=dict(
        in_channels=[64, 512],
        out_channels=512),
    reliability_fusion=dict(
        img_channels=64,
        pts_channels=512,
        fused_channels=512),
    deformable_cross_attention=dict(
        pts_channels=512))

fp16 = dict(loss_scale='dynamic')

# spconv 2.0 cannot autotune this R50 sparse encoder in FP16 eval mode.
# Train in FP16 and run validation separately in FP32.
evaluation = dict(interval=1, start=49)

load_from = 'ckpts/effocc_fusion_r50.pth'
resume_from = None
work_dir = 'work_dirs/oup_occ_fusion_r50_complete_version_mIoU_54_14'
