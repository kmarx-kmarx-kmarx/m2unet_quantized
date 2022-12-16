
# Installation
```bash
conda create -n m2unet python=3.8
conda install  cudatoolkit=11.1 -c nvidia
# You may need to change the following line based on https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy scipy imageio scikit-image albumentations bioimageio.core

```

# Usage

```bash
conda activate m2unet 
python test_native_speed.py
```