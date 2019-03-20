# model settings
input_size = 512
model = dict(
    type='SingleStageDetector',
    pretrained='snapshots/mobilenet_v2.pth.tar',
    backbone=dict(
        type='SSDMobilenetV2',
        input_size=input_size
        ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(576, 1280, 512, 256, 256, 128),
        num_classes=2,
        anchor_strides=(16, 32, 64, 128, 256, 512),
        basesize_ratio_range=(0.1, 0.9),
        anchor_ratios=([1.2], [1.2, 1.3], [1.2, 1.3], [1.2, 1.3], [1.2], [1.2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        depthwise_heads=True,
        adaptive_loss_weighting=True))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    use_giou=True,
    use_focal=False,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'WIDERDataset'
data_root = 'data/WIDER/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[1./255, 1./255, 1./255], to_rgb=True)
data = dict(
    imgs_per_gpu=25,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'train.txt',
            ],
            img_prefix=[data_root + 'WIDER_train/'],
            img_scale=(512, 512),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 2)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=1e-1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[15, 19, 24])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        #dict(type='TextLoggerHook'),
         dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd512_wider'
load_from = None
resume_from = None
workflow = [('train', 1)]
