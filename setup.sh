git clone https://huggingface.co/KhangTruong/vit-rvsa
python extra_code.py
cd vit-rvsa
rm whole_vit.pth
wget -q https://huggingface.co/KhangTruong/vit-rvsa/resolve/main/whole_vit.pth -O whole_vit.pth
wget -q https://huggingface.co/KhangTruong/vit-rvsa/resolve/main/checkpoint.pth -O checkpoint.pth
pip install -q numpy==1.26.4 flask einops timm Pillow
pip install -q --upgrade opencv-python
# --- cell 1 ---
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install -q mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# --- cell 3 ---
pip install -q "mmdet>=3.0.0rc6,<3.1.0"
pip install -q numpy==1.26.4
git clone https://github.com/open-mmlab/mmrotate -b 1.x
cd mmrotate
pip install -q -e .
cd ..
git clone https://github.com/ViTAE-Transformer/MTP
cp -r ./MTP/RS_Tasks_Finetune/Rotated_Detection/mmrotate1.x/mmrotate/* ./mmrotate/mmrotate
python install.py --root .
cd mmrotate
