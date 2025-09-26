import requests
import zipfile
import io
import os

# Server endpoint
url = "http://localhost:8000/predict"

# Path to your input image
input_path = "input.jpg"
output_image_path = "output.jpg"
output_json_path = "output.json"

# === Choose mode ===
return_json = True   # set to False for image only

params = {"json": "1"} if return_json else {}

with open(input_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, params=params)

if response.status_code == 200:
    if return_json:
        # Expecting a zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(".")  # extract into current folder
        print("✅ Extracted results:")
        for name in z.namelist():
            print(f" - {name}")
    else:
        # Save image directly
        with open(output_image_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Output image saved to {output_image_path}")
else:
    print("❌ Error:", response.status_code, response.text)
