_base_ = '../r3d_18_ucf101/finetune_ucf101.py'

work_dir = './output/ctp/r2plus1d_18_ucf101/ft_ssl_benchmark_ctp_ckpt_on_val_set/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='/scratch-shared/fmthoker/ssl_benchmark/checkpoints/CTP/Kinetics/pretext_checkpoint/r2p1d18_ctp_k400_epoch_90.pth',
    ),
)
