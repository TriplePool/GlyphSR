conda create -n tir python=3.9 -y
conda activate tir
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install wandb
pip install Cython
pip install -r requirements.txt