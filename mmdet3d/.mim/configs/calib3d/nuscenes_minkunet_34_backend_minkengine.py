_base_ = ['/cpfs01/user/konglingdong/models/mmdetection3d/configs/_base_/default_runtime.py']

# For nuScenes we usually do 16-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.

dataset_type = 'NuScenesSegDataset'
data_root = '/cpfs01/user/konglingdong/data/sets/nuscenes'
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]
labels_map = {
    0:  16,
    1:  16,
    2:  6,
    3:  6,
    4:  6,
    5:  16,
    6:  6,
    7:  16,
    8:  16,
    9:  0,
    10: 16,
    11: 16,
    12: 7,
    13: 16,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 16,
    20: 16,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 16,
    30: 15,
    31: 16
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31,
)

input_modality = dict(use_lidar=True, use_camera=False)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    img='',
    pts_semantic_mask='lidarseg/v1.0-trainval',
)


backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args,
    ),
    dict(
        type='PointSegClassMapping',
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask'],
    ),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args,
    ),
    dict(
        type='PointSegClassMapping',
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask'],
    ),
]


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/cpfs01/user/konglingdong/data/sets/nuscenes/nuscenes_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        serialize_data=True,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/cpfs01/user/konglingdong/data/sets/nuscenes/nuscenes_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        serialize_data=True,
        test_mode=True,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SegMetric'),
    dict(type='ECEMetric', file_name='minkunet.pkl'),
]
test_evaluator = val_evaluator


model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=True,  ##
        max_voxels=None,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.1, 0.1, 0.1],
            max_voxels=(-1, -1),
        ),
    ),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 3, 4, 6],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='minkowski',
    ),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96,
        num_classes=16,
        batch_first=True,
        dropout_ratio=0,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            avg_non_ignore=True,
        ),
        ignore_index=16,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

lr = 0.008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6,
    ),
)

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=120000,  # 120000 iters for 8xb2
        by_epoch=False,
        eta_max=0.01,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, save_best='miou', greater_keys='miou', interval=2000, max_keep_ckpts=1,
    )
)

log_processor = dict(type='LogProcessor', by_epoch=False)