import os
import subprocess
import shutil


os.chdir('vit-rvsa/mmrotate')

def predictor(directory):
    # directory = '../data/airplane/airplane_001.jpg'
    subprocess.run(['python', '../extract_feature.py', '-n', directory])
    return 'visualized_result.jpg'
