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

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_CLASSES = 2
MODEL_PATH = 'model.pt'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    logger.info("Loading model for app1...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
    else:
        model = state_dict

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully in app1!")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error("Please ensure 'model.pt' is in the same directory")
    raise

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def is_valid_skin_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = list(image.getdata())
        if len(set(img_array[:100])) < 10:
            return False
        width, height = image.size
        if width < 50 or height < 50:
            return False
        return True
    except:
        return False

@app.route('/')
def index():
    if os.path.exists('index1.html'):
        return send_file('index1.html')
    else:
        return jsonify({
            'message': 'Skin Cancer Detection API (app1)',
            'version': '1.0',
            'status': 'running',
            'endpoints': {
                '/': 'Serves web interface',
                '/health': 'Health check (GET)',
                '/predict': 'Image prediction (POST)'
            }
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

        logger.info(f"Processing image (app1): {image_file.filename}")

        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        if not is_valid_skin_image(image):
            logger.info("Image doesn't appear to be a valid skin lesion image")
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
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        cancer_percentage = float(probabilities[0][1].item() * 100)

        max_prob = float(torch.max(probabilities).item())
        if max_prob < 0.55:
            logger.info("Low confidence prediction - possibly invalid image")
            cancer_percentage = 0.0
            message = 'Unable to confidently analyze this image. Please ensure it is a clear photo of a skin lesion.'
        else:
            message = 'Analysis complete'

        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'cancer_percentage': round(cancer_percentage, 2),
            'message': message,
            'disclaimer': 'This is a preliminary screening tool. Always consult a qualified healthcare professional for diagnosis.'
        }

        logger.info("="*60)
        logger.info("PREDICTION RESULT (app1)")
        logger.info(f"Cancer Percentage: {cancer_percentage:.2f}%")
        logger.info("="*60)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during image analysis'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
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
    print("🔬 SKIN CANCER DETECTION SYSTEM (app1)")
    print("="*60)
    print(f"📱 Device: {device}")
    print(f"🧠 Model: Vision Transformer (ViT)")
    print(f"🌐 Server: http://localhost:5001")
    print(f"📄 Interface: Open index1.html in your browser")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)


