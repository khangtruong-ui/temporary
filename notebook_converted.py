"""Converted from Re_remote_Object_detection.ipynb - main logic in `run_inference(image_bytes)`"""
import io, os, sys
import PIL  # auto-added

# --- cell 1 ---
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install -q mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# --- cell 3 ---
pip install -q "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/open-mmlab/mmrotate -b 1.x
cd mmrotate
pip install -q -e .
cd ..
git clone https://github.com/ViTAE-Transformer/MTP
cp -r ./MTP/RS_Tasks_Finetune/Rotated_Detection/mmrotate1.x/mmrotate/* ./mmrotate/mmrotate
git clone https://huggingface.co/datasets/KhangTruong/NWPU-Caption
tar -xf NWPU-Caption/02_NWPU_RESISC45.tar
tar -xf NWPU-Caption/02_NWPU_caption.tar
mv 02_NWPU_RESISC45 data
mv 02_NWPU_caption data
python install.py --root .
cd mmrotate

# --- cell 5 ---
from PIL import Image
for i in range(1, 11):
    directory = f'/content/Images/{i:02d}.png'
    # directory = '../data/airplane/airplane_001.jpg'
