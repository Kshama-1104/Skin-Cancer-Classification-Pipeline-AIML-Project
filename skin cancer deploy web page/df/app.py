# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms
import timm
import io
import os
from datetime import datetime
import logging
import gdown
import base64
import shutil

app = Flask(__name__)
CORS(app)

# ----------------------------
# logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 2))
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")

# <-- YOUR GOOGLE DRIVE FILE ID (hard-coded here per your request) -->
DRIVE_FILE_ID = "1TWnhVytreEc2gnNmZr_y9iz_eZ4DrgjJ"

# Optional: Hugging Face repo fallback (not used here)
HF_HUB_REPO = os.getenv("HF_HUB_REPO")
HF_FILENAME = os.getenv("HF_FILENAME", "model.pt")

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ----------------------------
# helper: download model if needed
# ----------------------------
def download_model_from_drive(file_id: str, dest: str):
    """
    Download a file from Google Drive using gdown.
    Accepts the plain file_id (the portion between /d/ and /view).
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        logger.info(f"Model already present at {dest} ({os.path.getsize(dest)} bytes)")
        return True

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading model from Google Drive id={file_id} to {dest} ...")
    try:
        gdown.download(url, dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info("Download complete.")
            return True
        else:
            logger.error("Download failed: file not found or zero bytes after gdown")
            return False
    except Exception as e:
        logger.exception(f"Exception while downloading from drive: {e}")
        return False

def download_model_from_hf(repo: str, filename: str, dest: str):
    """
    Optional: download from Hugging Face Hub using huggingface_hub.hf_hub_download
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        logger.warning("huggingface_hub not installed or available.")
        return False

    try:
        logger.info(f"Downloading {filename} from HF repo {repo} to {dest} ...")
        path = hf_hub_download(repo_id=repo, filename=filename)
        shutil.copyfile(path, dest)
        logger.info("HF download complete.")
        return True
    except Exception as e:
        logger.exception(f"Error downloading from Hugging Face Hub: {e}")
        return False

# ----------------------------
# model loading utility
# ----------------------------
def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v
    return new_state

def robust_load_model(model_path, num_classes=NUM_CLASSES):
    """
    Attempts to load the model robustly:
    - Instantiates a ViT model with timm, then loads state_dict (handles nested keys)
    - If the file itself is a serialized model object, returns it directly.
    """
    logger.info(f"Attempting to load model from {model_path}")
    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, dict):
        candidate = None
        if 'model_state_dict' in obj:
            candidate = obj['model_state_dict']
        elif 'state_dict' in obj:
            candidate = obj['state_dict']
        else:
            candidate = obj

        if isinstance(candidate, dict):
            logger.info("Found state_dict-like object; creating model architecture and loading weights.")
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            candidate = strip_module_prefix(candidate)
            model.load_state_dict(candidate, strict=False)
            return model
        else:
            logger.info("state_dict not dict; treating loaded object as model instance.")
            return candidate
    else:
        logger.info("Loaded object is not a dict - treating as model instance.")
        return obj

# ----------------------------
# Ensure model exists: download if needed
# ----------------------------
if not os.path.exists(MODEL_PATH):
    downloaded = False
    if DRIVE_FILE_ID:
        downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_PATH)
    if (not downloaded) and HF_HUB_REPO:
        downloaded = download_model_from_hf(HF_HUB_REPO, HF_FILENAME, MODEL_PATH)
    if not downloaded:
        logger.warning(f"Model file {MODEL_PATH} not found and no download succeeded. "
                       "Ensure MODEL_PATH or DRIVE_FILE_ID / HF_HUB_REPO env var is set.")

# ----------------------------
# Load the model
# ----------------------------
model = None
model_loaded = False
try:
    if os.path.exists(MODEL_PATH):
        model = robust_load_model(MODEL_PATH, num_classes=NUM_CLASSES)
        if isinstance(model, torch.nn.Module):
            model.to(device)
            model.eval()
            model_loaded = True
            logger.info("✅ Model loaded and set to eval mode.")
        else:
            logger.warning("Loaded object is not a nn.Module. Model may be incompatible.")
    else:
        logger.warning("No model file found at MODEL_PATH; skipping load.")
except Exception as e:
    logger.exception(f"Exception while loading the model: {e}")
    model = None
    model_loaded = False

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Basic image validity check
# ----------------------------
def is_valid_skin_image(image: Image.Image) -> bool:
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        if width < 50 or height < 50:
            return False

        small = image.resize((32, 32))
        pixels = list(small.getdata())
        unique_colors = len(set(pixels))
        if unique_colors < 6:
            return False

        return True
    except Exception as e:
        logger.exception(f"Exception in is_valid_skin_image: {e}")
        return False

# ----------------------------
# Endpoints
# ----------------------------
@app.route('/')
def index():
    if os.path.exists('index.html'):
        return send_file('index.html')
    else:
        return jsonify({
            'message': 'Skin Cancer Detection API',
            'version': '3.2',
            'status': 'running',
            'model_loaded': model_loaded,
            'endpoints': {
                '/': 'Serves web interface (index.html)',
                '/health': 'Health check (GET)',
                '/predict': 'Image prediction (POST) - form-data "image" or JSON {"image_base64": "..."}'
            }
        })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or model is None:
        return jsonify({'success': False, 'error': 'Model not loaded on server'}), 500

    try:
        image = None
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            allowed_extensions = {'png', 'jpg', 'jpeg'}
            file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
            if file_ext not in allowed_extensions:
                return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get('image_base64') or data.get('image')
            if not b64:
                return jsonify({'error': 'No image provided. Use multipart form field "image" or JSON "image_base64"'}), 400

            if isinstance(b64, str) and b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]

            try:
                image_bytes = base64.b64decode(b64)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception:
                return jsonify({'error': 'Invalid base64 image data'}), 400

        if not is_valid_skin_image(image):
            logger.info("Image failed basic validity checks.")
            return jsonify({
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'cancer_percentage': 0.0,
                'message': 'Image does not appear to be a valid skin lesion image',
                'disclaimer': 'This is a preliminary screening tool. Always consult a qualified healthcare professional for diagnosis.'
            })

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs

            if logits.dim() == 1:
                probs_malignant = torch.sigmoid(logits)
                if probs_malignant.numel() == 1:
                    cancer_prob = float(probs_malignant.item())
                    probabilities = torch.tensor([[1 - cancer_prob, cancer_prob]])
                else:
                    probabilities = probs_malignant
            else:
                probabilities = torch.nn.functional.softmax(logits, dim=1)

        if probabilities.dim() == 1:
            # shape (C,) -> make batch dim
            probabilities = probabilities.unsqueeze(0)

        if probabilities.shape[1] < 2:
            logger.warning("Model output has fewer than 2 classes; interpreting max as cancer prob fallback.")
            cancer_percentage = float(probabilities[0].max().item() * 100)
        else:
            cancer_percentage = float(probabilities[0][1].item() * 100)

        max_prob = float(torch.max(probabilities).item())
        if max_prob < 0.55:
            message = 'Unable to confidently analyze this image. Please ensure it is a clear photo of a skin lesion.'
            logger.info("Low confidence (%0.3f). Marking as unable to analyze.", max_prob)
            cancer_percentage = 0.0
        else:
            message = 'Analysis complete'

        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'cancer_percentage': round(cancer_percentage, 2),
            'message': message,
            'disclaimer': 'This is a preliminary screening tool. Always consult a qualified healthcare professional for diagnosis.'
        }

        logger.info("PREDICTION: cancer_percentage=%0.2f message=%s", cancer_percentage, message)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during image analysis'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device),
        'model_type': 'Vision Transformer (ViT)',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 SKIN CANCER DETECTION SYSTEM")
    print("="*60)
    print(f"📱 Device: {device}")
    print(f"🧠 Model loaded: {model_loaded}")
    print(f"🌐 Server: http://localhost:{os.getenv('PORT', 5000)}")
    print(f"📄 Interface: Open index.html in your browser (if present)")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
