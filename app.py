import os
import uuid
import sys
import types
from flask import Flask, render_template, request, redirect, url_for, flash
import cv2

import torchvision.transforms.functional as F
functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor

from ui_inference import LLFSRPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.secret_key = "face_super_resolution_secret"

pipeline = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pipeline():
    global pipeline
    if pipeline is None:
        enhancer_ckpt = os.path.join(BASE_DIR, "llfsr_enhancer_v5.pth")
        gfpgan_ckpt = os.path.join(BASE_DIR, "GFPGANv1.4.pth")
        if not os.path.exists(enhancer_ckpt) or not os.path.exists(gfpgan_ckpt):
            raise FileNotFoundError(
                f"Required model checkpoint missing. Expected files:\n"
                f"  {enhancer_ckpt}\n  {gfpgan_ckpt}"
            )
        pipeline = LLFSRPipeline(enhancer_ckpt, gfpgan_ckpt)
    return pipeline


@app.route("/", methods=["GET", "POST"])
def index():
    input_image = None
    output_image = None
    status = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            flash("Please select a low-light face image to upload.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file format. Use PNG, JPG, or JPEG.", "error")
            return redirect(request.url)

        filename = f"{uuid.uuid4().hex}_input.{file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        source_image = cv2.imread(upload_path)
        if source_image is None:
            flash("Could not read the uploaded image. Try a different file.", "error")
            return redirect(request.url)

        try:
            pipeline_obj = get_pipeline()
            result_bgr = pipeline_obj.enhance(source_image)
        except Exception as exc:
            flash(f"Processing failed: {exc}", "error")
            return redirect(request.url)

        output_filename = f"{uuid.uuid4().hex}_result.png"
        output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
        cv2.imwrite(output_path, result_bgr)

        input_image = url_for("static", filename=f"uploads/{filename}")
        output_image = url_for("static", filename=f"results/{output_filename}")
        
        input_brightness = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).mean()
        output_brightness = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).mean()
        brightness_improvement = ((output_brightness - input_brightness) / input_brightness * 100)
        
        status = ".1f"

    return render_template(
        "index.html",
        input_image=input_image,
        output_image=output_image,
        status=status,
    )


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
