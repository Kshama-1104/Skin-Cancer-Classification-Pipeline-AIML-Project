import io
import os
from typing import List, Tuple
import json

from PIL import Image
import torch
from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__, static_folder="static", static_url_path="/static")


def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Prefer TorchScript models for deployment
    try:
        scripted = torch.jit.load(model_path, map_location=torch.device("cpu"))
        scripted.eval()
        return scripted
    except Exception:
        pass

    obj = torch.load(model_path, map_location=torch.device("cpu"))

    # If it's a plain state_dict (OrderedDict), we cannot run inference without the model class
    if isinstance(obj, dict) and not hasattr(obj, "forward"):
        from collections import OrderedDict

        if isinstance(obj, OrderedDict) or (
            "state_dict" in obj and isinstance(obj["state_dict"], OrderedDict)
        ):
            raise RuntimeError(
                "Loaded checkpoint is a state_dict (parameters only). "
                "Provide the model architecture code to rebuild the model and load the state_dict, "
                "or export a TorchScript model with torch.jit.script/trace and use that .pt file."
            )

    # Otherwise assume it's a full nn.Module checkpoint
    model = obj
    if hasattr(model, "eval"):
        model.eval()
    return model


MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))

def get_normalization() -> Tuple[List[float], List[float]]:
    mean = os.environ.get("MEAN")
    std = os.environ.get("STD")
    if mean and std:
        mean_vals = [float(x) for x in mean.split(",")]
        std_vals = [float(x) for x in std.split(",")]
        if len(mean_vals) == 3 and len(std_vals) == 3:
            return mean_vals, std_vals
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

NORM_MEAN, NORM_STD = get_normalization()


def load_labels() -> List[str]:
    labels_path = os.environ.get("LABELS_PATH", "labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass
    return []

LABELS = load_labels()
MODEL = load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Build a tensor directly from the PIL raw bytes to avoid numpy dependency
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    float_tensor = byte_tensor.float().div(255.0)
    float_tensor = float_tensor.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1)
    # Normalize
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    float_tensor = (float_tensor - mean) / std
    float_tensor = float_tensor.unsqueeze(0)  # add batch dim NCHW
    return float_tensor


def postprocess_output(output: torch.Tensor) -> Tuple[int, float]:
    if output.ndim == 2 and output.size(0) == 1:
        output = output[0]
    if output.ndim == 1:
        probs = torch.softmax(output, dim=0)
        score, idx = torch.max(probs, dim=0)
        return int(idx.item()), float(score.item())
    raise ValueError("Model output shape not supported for simple argmax.")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded with field name 'image'"}), 400
    file = request.files["image"]
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = MODEL(input_tensor)
        pred_idx, confidence = postprocess_output(output)
        response = {"predicted_index": pred_idx, "confidence": confidence}
        if LABELS and 0 <= pred_idx < len(LABELS):
            response["label"] = LABELS[pred_idx]
        return jsonify(response)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


