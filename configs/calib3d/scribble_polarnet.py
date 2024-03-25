_base_ = ['../_base_/default_runtime.py']

# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'SemanticKittiDataset'
data_root = '/data/sets/semantickitti'

class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,   # "unlabeled"
    1: 19,   # "outlier" mapped to "unlabeled" --------------mapped
    10: 0,   # "car"
    11: 1,   # "bicycle"
    13: 4,   # "bus" mapped to "other-vehicle" --------------mapped
    15: 2,   # "motorcycle"
    16: 4,   # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 3,   # "truck"
    20: 4,   # "other-vehicle"
    30: 5,   # "person"
    31: 6,   # "bicyclist"
    32: 7,   # "motorcyclist"
    40: 8,   # "road"
    44: 9,   # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 8,   # "lane-marking" to "road" ---------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 19,  # "other-object" to "unlabeled" ----------------mapped
    252: 0,  # "moving-car" to "car" ------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 5,  # "moving-person" to "person" ------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 3,  # "moving-truck" to "truck" --------------------mapped
    259: 4   # "moving-other"-vehicle to "other-vehicle"-----mapped
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259)

input_modality = dict(use_lidar=True, use_camera=False)


backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(
        type='PointSegClassMapping',
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask'],
    )
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args,
    ),
    dict(
        type='PointSegClassMapping',
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask'],
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True, seed=0),
    # sampler=dict(type='DefaultSampler', shuffle=True),  # for calibration
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/sets/semantickitti/semantickitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        backend_args=backend_args,
        scribble=True,  ## scribblekitti
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/sets/semantickitti/semantickitti_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=True,
        backend_args=backend_args,
        scribble=False,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

grid_shape = [480, 360, 32]
point_cloud_range = [3, -3.1415926, -3, 50, 3.1415926, 1.5]
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
            max_voxels=-1,
        )
    ),
    voxel_encoder=dict(
        type='SegVFE',
        in_channels=6,
        feat_channels=[64, 128, 256, 512],
        with_voxel_center=True,
        grid_shape=grid_shape,
        point_cloud_range=point_cloud_range,
        feat_compression=32,
        height_pooling=True,
    ),
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
        dropout_ratio=0.5,
    ),
    decode_head=dict(
        type='PolarHead',
        channels=64,
        num_classes=20,
        dropout_ratio=0,
        height=32,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
        conv_seg_kernel_size=1,
        ignore_index=19,
    ),
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
        total_steps=45000,  # 45000 iters for 8xb2
        by_epoch=False,
        eta_max=0.01,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=45000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
# env_cfg = dict(cudnn_benchmark=True)

default_hooks = dict(checkpoint=dict(by_epoch=False, save_best='miou', greater_keys='miou'))
log_processor = dict(by_epoch=False)