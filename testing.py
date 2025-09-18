import requests

# Server endpoint
url = "http://localhost:8000/predict"

# Path to your input image
input_path = "input.jpg"
output_path = "output.jpg"

# Send request
with open(input_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Save result
if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Output image saved to {output_path}")
else:
    print("❌ Error:", response.status_code, response.text)
