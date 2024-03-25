_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.CENet.cenet'], allow_failed_imports=False,
)


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
        type='PointSample',
        num_points=0.9,
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type='SemkittiRangeView',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=19,
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('proj_x', 'proj_y', 'proj_range', 'unproj_range'),  ### for calibration
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
        type='SemkittiRangeView',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=19,
    ),
    dict(
        type='Pack3DDetInputs',
        # keys=['img'],
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('proj_x', 'proj_y', 'proj_range', 'unproj_range'),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/sets/semantickitti/semantickitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler', shuffle=False,
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
    ),
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='SegMetric'),
    dict(type='ECEMetric', file_name='semkitti_salsanext.pkl'),
]
test_evaluator = val_evaluator


model = dict(
    type='RangeImageSegmentor',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
    ),
    backbone=dict(
        type='SalsaNext',
        in_channels=5,
    ),
    decode_head=dict(
        type='RangeImageHead',
        channels=32,
        num_classes=20,
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ),
        loss_lovasz=dict(
            type='LovaszLoss',
            loss_weight=1.0,
            reduction='none',
        ),
        conv_seg_kernel_size=1,
        ignore_index=19,
    ),
    train_cfg=None,
    test_cfg=dict(use_knn=False, knn=7, search=7, sigma=1.0, cutoff=2.0),
)

# optimizer
# This schedule is mainly used on Semantickitti dataset in segmentation task
lr = 0.008
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-6,
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

# runtime settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=45000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, save_best='miou', greater_keys='miou', interval=1000, max_keep_ckpts=1,
    )
)

log_processor = dict(by_epoch=False)