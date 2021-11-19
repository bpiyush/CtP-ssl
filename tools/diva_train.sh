# run training for a given config
echo "$(dirname "$0")/train_net.py"
python $(dirname "$0")/train_net.py \
    --cfg configs/ctp/r2plus1d_18_ucf101/finetune_ucf101.py \
    --work_dir /home/pbagad/expts/ctp-ssl/ \
    --data_dir /ssd/pbagad/datasets/ \
    --freeze_backbone \
    --ckpt /home/pbagad/projects/ssl_benchmark/checkpoints/CTP/Kinetics/pretext_checkpoint/r2p1d18_ctp_k400_epoch_90.pth