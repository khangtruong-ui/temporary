import os
import subprocess
import shutil

# This assumes the directory 'vit-rvsa/mmrotate' exists
os.chdir('vit-rvsa/mmrotate')

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
    
    try:
        # Save the image bytes to the temporary file
        with open(temp_img_path, 'wb') as f:
            f.write(img_bytes)
        
        # Process the temporary image file using the subprocess
        subprocess.run(
            ['python', '../extract_feature.py', '-n', temp_img_path],
            check=True  # Raise an exception if the command fails
        )
        
        # The subprocess should generate the 'visualized_result.jpg' file
        return 'visualized_result.jpg'
        
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
