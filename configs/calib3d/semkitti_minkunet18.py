_base_ = ['../_base_/default_runtime.py']


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
    classes=class_names, seg_label_mapping=labels_map, max_label=259,
)

input_modality = dict(use_lidar=True, use_camera=False)

backend_args = None

train_pipeline = [
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
        keys=['points'],
    ),
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
        scribble=False,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
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
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SegMetric'),
    dict(type='ECEMetric', file_name='semkitti_minkunet18.pkl'),
]
test_evaluator = val_evaluator


model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=True,
        max_voxels=80000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1),
        ),
    ),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='spconv',
    ),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96,
        num_classes=19,
        batch_first=False,
        dropout_ratio=0,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            avg_non_ignore=True,
        ),
        ignore_index=19,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

lr = 0.008
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6,
    ),
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

env_cfg = dict(cudnn_benchmark=True)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, save_best='miou', greater_keys='miou', interval=1000, max_keep_ckpts=1,
    )
)

log_processor = dict(by_epoch=False)