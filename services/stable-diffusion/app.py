from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import io
import base64
import uuid

app = Flask(__name__)
CORS(app)

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Disable safety checker (for dev only)
pipe.safety_checker = lambda images, **kwargs: (images, False)

# Image generation endpoint
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    width = data.get("width", 512)
    height = data.get("height", 512)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        image = pipe(prompt, height=height, width=width).images[0]
        output_dir = os.path.join(os.path.dirname(__file__), "generated")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)

        # Optional base64 encoding (for previews if needed)
        # buffered = io.BytesIO()
        # image.save(buffered, format="PNG")
        # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "imageUrl": f"http://127.0.0.1:5002/generated/{filename}"
            # "base64_image": img_str
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve generated images
@app.route('/generated/<filename>')
def serve_image(filename):
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
