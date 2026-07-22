"""
Gradio UI for Hugging Face Spaces (free ZeroGPU / Gradio path).
Docker Spaces are paid — use this file as the Space entrypoint instead.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_HOME', os.path.join(os.path.dirname(__file__), '.keras'))

import cv2
import gradio as gr
import numpy as np

# Loads models once (same pipeline as Flask app)
from app import run_prediction, load_custom_face_index


def recognize(image, method, classifier, fusion):
    if image is None:
        return "Please upload or capture an image.", ""

    # Gradio images are RGB; OpenCV expects BGR
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = run_prediction(
        bgr,
        method=method,
        classifier=classifier,
        activation='softmax',
        fusion=fusion,
    )

    label = result.get('label', 'Unknown')
    conf = result.get('confidence', 'N/A')
    level = result.get('confidence_level', '')
    summary = f"**{label}**\n\nConfidence: {conf} ({level})"

    details = (
        f"Method: {result.get('method', method)}\n"
        f"Model: {result.get('model_used', classifier)}\n"
        f"Fusion: {result.get('fusion_type', fusion)}\n"
        f"Activation: {result.get('activation', 'softmax')}"
    )
    return summary, details


def list_registered():
    index = load_custom_face_index()
    faces = index.get('registered_faces', {})
    if not faces:
        return "No custom faces registered yet."
    lines = [f"- {f['name']} (id={f['id']}, images={f.get('num_images', '?')})"
             for f in faces.values()]
    return "**Registered faces**\n\n" + "\n".join(lines)


with gr.Blocks(title="Hybrid Face Recognition System") as demo:
    gr.Markdown(
        """
        # Hybrid Face Recognition System
        SSIEMS Parbhani — M.Tech Research Project  
        SIFT · HOG · Gabor · Canny + CNN / MobileNetV2 / SVM
        """
    )

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Upload or webcam", sources=["upload", "webcam"])
            method = gr.Dropdown(
                ["hybrid", "sift", "hog", "gabor", "canny"],
                value="hybrid",
                label="Feature method",
            )
            classifier = gr.Dropdown(
                ["custom", "mobilenetv2", "svm_rbf", "svm_linear"],
                value="custom",
                label="Classifier",
            )
            fusion = gr.Dropdown(
                ["full", "research"],
                value="full",
                label="Fusion",
            )
            btn = gr.Button("Recognize", variant="primary")
        with gr.Column():
            out_summary = gr.Markdown(label="Result")
            out_details = gr.Textbox(label="Details", lines=5)
            registered = gr.Markdown(list_registered())

    btn.click(
        recognize,
        inputs=[image, method, classifier, fusion],
        outputs=[out_summary, out_details],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
