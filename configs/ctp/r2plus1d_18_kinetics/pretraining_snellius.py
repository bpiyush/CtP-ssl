_base_ = '../r3d_18_kinetics/pretraining.py'

work_dir = './output/ctp/r2plus1d_18_kinetics/pretraining_snellius/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    )
)

data = dict(
    videos_per_gpu=16,  # total batch size is 4Gpus*16 == 64
    workers_per_gpu=16,
    train=dict(
        data_source=dict(
            ann_file='kinetics400/annotations/train_split_1.json',
        ),
        backend=dict(
            zip_fmt='kinetics400/zips/{}.zip',
        )
    )
)