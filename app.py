
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from diffusers import StableDiffusionPipeline
import torch
import os

app = Flask(__name__, static_folder="static", template_folder="static")

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load the pipeline once at startup
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/")
def index():
    # Serve the static index.html
    return send_from_directory(app.static_folder, "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400
    try:
        image = pipe(prompt).images[0]
        filename = f"generated_image.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        image_url = f"/results/{filename}"
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/results/<filename>")
def get_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)