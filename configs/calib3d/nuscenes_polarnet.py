_base_ = ['../_base_/default_runtime.py']

# For nuScenes we usually do 16-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.

dataset_type = 'NuScenesSegDataset'
data_root = '/data/sets/nuscenes/'
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
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
    classes=class_names, seg_label_mapping=labels_map, max_label=31)

input_modality = dict(use_lidar=True, use_camera=False)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    img='',
    pts_semantic_mask='lidarseg/v1.0-trainval')


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
    )
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
        keys=['points'],
    )
]

tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[pcd_rotate_range, pcd_rotate_range],
                            scale_ratio_range=[
                                pcd_scale_factor, pcd_scale_factor
                            ],
                            translation_std=[0, 0, 0])
                        for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                        for pcd_scale_factor in [0.95, 1.0, 1.05]
                    ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/sets/nuscenes/nus_info/nuscenes_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        backend_args=backend_args,
    )
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
        ann_file='/data/sets/nuscenes/nus_info/nuscenes_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        test_mode=True,
        backend_args=backend_args,
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SegMetric'),
    dict(type='ECEMetric', file_name='nuscenes_polarnet.pkl'),
]

test_evaluator = val_evaluator


# tta_model = dict(type='Seg3DTTAModel')

grid_shape = [480, 360, 32]
point_cloud_range = [0, -3.1415926, -5, 50, 3.1415926, 3]
model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        in_channels=6,
        feat_channels=[64, 128, 256, 512],
        with_voxel_center=True,
        grid_shape=grid_shape,
        point_cloud_range=point_cloud_range,
        feat_compression=32,
        height_pooling=True),
    backbone=dict(
        type='PolarUNet',
        in_channels=32,
        output_shape=(480, 360),
        stem_channels=64,
        num_stages=4,
        down_channels=(128, 256, 512, 512),
        up_channels=(256, 128, 64, 64),
        pre_norm=True,
        circular_padding=True,
        use_dropblock=True,
        dropout_ratio=0.5),
    decode_head=dict(
        type='PolarHead',
        channels=64,
        num_classes=17,
        dropout_ratio=0,
        height=32,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
        conv_seg_kernel_size=1,
        ignore_index=16),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

lr = 0.008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-6,
    )
)

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=80000,  # 80000 iters for 8xb2
        by_epoch=False,
        eta_max=0.01,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, save_best='miou', greater_keys='miou', interval=1000, max_keep_ckpts=1,
    )
)

log_processor = dict(type='LogProcessor', by_epoch=False)