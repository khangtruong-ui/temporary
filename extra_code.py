import os
import shutil

shutil.copy('../dior_run_config.py', '.')
shutil.move('../whole_vit.pth', '.')
shutil.move('../checkpoint.pth', '.')
