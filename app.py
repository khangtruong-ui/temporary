from flask import Flask, request, jsonify
import traceback
import core
import os
import json

app = Flask(__name__)

def find_inference_func(mod):
    for name in ("run_inference","predict","inference","process_image","main","predictor"):
        if hasattr(mod, name):
            return getattr(mod, name)
    return None

INFER = find_inference_func(core)

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok')

@app.route('/predict', methods=['POST'])
def predict():
    if INFER is None:
        return jsonify(error="No inference function found"), 500
    if 'file' not in request.files:
        return jsonify(error="No file part, use multipart form field named 'file'"), 400
    
    f = request.files['file']
    img_bytes = f.read()

    try:
        json_path = INFER(img_bytes)
        if json_path == 'error' or not os.path.exists(json_path):
            return jsonify(error=f"Failed to generate output.json from {json_path}"), 500

        with open(json_path, "r") as jf:
            data = json.load(jf)
        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
