_base_ = ['./oup_occ_fusion_r18_base_100%_seqs.py']

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
        lambda_u=1.0))

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

load_from = None
work_dir = 'work_dirs/oup_occ_fusion_r18_tuned_100%_seqs'
