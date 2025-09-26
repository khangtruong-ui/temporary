from flask import Flask, request, send_file, jsonify
from io import BytesIO
import traceback
import core
import os
import zipfile
import io

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
    return_json = request.args.get("json", "0") == "1"  # check query param ?json=1

    try:
        out = INFER(img_bytes, json_return=return_json)

        if isinstance(out, tuple):  # (json_path, image_path)
            json_path, image_path = out

            # Bundle JSON + Image into a zip in-memory
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                if os.path.exists(json_path):
                    zf.write(json_path, arcname="output.json")
                if os.path.exists(image_path):
                    zf.write(image_path, arcname="visualized_result.jpg")
            memory_file.seek(0)
            return send_file(memory_file, download_name="result.zip", as_attachment=True)

        elif isinstance(out, str):
            return send_file(out, mimetype='image/jpeg')

        elif isinstance(out, bytes):
            return send_file(BytesIO(out), mimetype='image/jpeg')

        else:
            return jsonify(result=str(type(out)))

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
