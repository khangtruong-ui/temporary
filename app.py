
from flask import Flask, request, send_file, jsonify
from io import BytesIO
import traceback
import core

app = Flask(__name__)

# Expect the converted notebook to expose a function `run_inference(image_bytes)` that returns bytes (processed image)
# If not found, we will try to look for a `main` or `predict` function in the module.

def find_inference_func(mod):
    for name in ("run_inference","predict","inference","process_image","main", "predictor"):
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
        return jsonify(error="No inference function found in notebook_converted module. Please implement run_inference(image_bytes)"), 500
    if 'file' not in request.files:
        return jsonify(error="No file part, use multipart form field named 'file'"), 400
    f = request.files['file']
    img_bytes = f.read()
    try:
        out = INFER(img_bytes)
        if isinstance(out, bytes):
            return send_file(BytesIO(out), mimetype='image/jpeg')
        elif isinstance(out, str):
            # maybe path
            return send_file(out, mimetype='image/jpeg')
        else:
            # try jsonify
            return jsonify(result=str(type(out)))
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
