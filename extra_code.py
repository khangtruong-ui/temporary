import os
import shutil

try:
    shutil.move('../whole_vit.pth', '.')
    shutil.move('../checkpoint.pth', '.')

except:
    pass
