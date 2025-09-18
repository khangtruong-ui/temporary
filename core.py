import os
import subprocess
import shutil
import sys
from PIL import Image

# This assumes the directory 'vit-rvsa/mmrotate' exists
os.chdir('vit-rvsa/mmrotate')

def resize_image(input_path: str, output_path: str, size: tuple = (800, 800)):
    """
    Reads an image from a given path, resizes it to the specified size,
    and saves it to a new path.

    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path where the resized image will be saved.
        size (tuple): A tuple containing the new width and height (e.g., (800, 800)).
    """
    if not os.path.exists(input_path):
        print(f"Error: The input file '{input_path}' does not exist.")
        return

    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Resize the image to the specified size.
            # Use img.resize() to force the new dimensions.
            # Use img.thumbnail() if you want to maintain the aspect ratio.
            resized_img = img.resize(size)

            # Save the resized image to the output path
            resized_img.save(output_path)
            print(f"Successfully resized '{input_path}' and saved it to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def predictor(img_bytes):
    """
    Saves image bytes to a temporary file, processes it, and then deletes the file.
    
    Args:
        img_bytes (bytes): The bytes data of the image.
    
    Returns:
        str: The name of the resulting image file, or an empty string if an error occurs.
    """
    # Create a temporary file with a unique name
    temp_img_path = 'temp_image.jpg'
    temp_secondary_path = 'temp.jpg'
    
    try:
        # Save the image bytes to the temporary file
        with open(temp_secondary_path, 'wb') as f:
            f.write(img_bytes)

        resize_image(temp_secondary_path, temp_img_path)
        print('============================== PROCESSING ===============================', file=sys.stderr)
        # Process the temporary image file using the subprocess
        subprocess.run(
            ['python', '../extract_feature.py', '-n', temp_img_path],
            stdout=sys.stderr,
            check=True,  # Raise an exception if the command fails
        )
        print('============================== DONE ===============================', file=sys.stderr)
        # The subprocess should generate the 'visualized_result.jpg' file
        return '../visualized_result.jpg'
        
    except FileNotFoundError as e:
        print(f"Error: The file or command was not found. {e}")
        return 'error'
    except subprocess.CalledProcessError as e:
        print(f"Error: Subprocess failed with exit code {e.returncode}")
        return 'error'
    finally:
        # Clean up: Delete the temporary image file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
