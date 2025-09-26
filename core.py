import os
import subprocess
import shutil
import sys
from PIL import Image

# This assumes the directory 'vit-rvsa/mmrotate' exists
os.chdir('vit-rvsa/mmrotate')

def resize_image(input_path: str, output_path: str, size: tuple = (800, 800)):
    if not os.path.exists(input_path):
        print(f"Error: The input file '{input_path}' does not exist.")
        return
    try:
        with Image.open(input_path) as img:
            resized_img = img.resize(size)
            resized_img.save(output_path)
            print(f"Successfully resized '{input_path}' and saved it to '{output_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def predictor(img_bytes, json_return: bool = False):
    """
    Saves image bytes to a temporary file, processes it, and then deletes the file.
    
    Args:
        img_bytes (bytes): The bytes data of the image.
        json_return (bool): If True, also return the JSON path.
    
    Returns:
        str | tuple: Image path (default), or (json_path, image_path) if json_return=True
    """
    temp_img_path = 'temp_image.jpg'
    temp_secondary_path = 'temp.jpg'
    
    try:
        with open(temp_secondary_path, 'wb') as f:
            f.write(img_bytes)

        resize_image(temp_secondary_path, temp_img_path)
        print('============================== PROCESSING ===============================', file=sys.stderr)

        subprocess.run(
            ['python', '../extract_feature.py', '-n', temp_img_path],
            stdout=sys.stderr,
            check=True,
        )

        print('============================== DONE ===============================', file=sys.stderr)

        image_path = 'vit-rvsa/mmrotate/visualized_result.jpg'
        json_path = 'vit-rvsa/mmrotate/output.json'

        if json_return:
            return (json_path, image_path)
        return json_path

    except FileNotFoundError as e:
        print(f"Error: The file or command was not found. {e}")
        return 'error'
    except subprocess.CalledProcessError as e:
        print(f"Error: Subprocess failed with exit code {e.returncode}")
        return 'error'
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists(temp_secondary_path):
            os.remove(temp_secondary_path)
