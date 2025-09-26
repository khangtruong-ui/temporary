import requests
import json

# Server endpoint
url = "http://localhost:8000/predict"

# Path to your input image
input_path = "input.jpg"

with open(input_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    try:
        data = response.json()
        print("✅ JSON result:")
        print(json.dumps(data, indent=2))
    except Exception:
        print("❌ Failed to parse JSON:", response.text)
else:
    print("❌ Error:", response.status_code, response.text)
