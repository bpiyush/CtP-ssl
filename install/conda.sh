conda create -n ctp
conda activate ctp

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html
python -c "import torch; print('Torch Version: ', torch.__version__)"
python -c "import torch; x = torch.randn(3, 4).cuda()"

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
python -c "import mmcv; print('MMCV version: ', mmcv.__version__)"

pip install tensorboard
