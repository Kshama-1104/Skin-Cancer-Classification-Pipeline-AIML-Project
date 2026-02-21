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
CORS(app)  # Enable CORS for frontend communication

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_CLASSES = 2
CLASS_LABELS = ['benign', 'malignant']
MODEL_PATH = 'model.pt'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    logger.info("Loading model...")
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
    logger.info("✅ Model loaded successfully!")
    
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    logger.error("Please ensure 'model.pt' is in the same directory")
    raise

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def get_risk_level(predicted_class, benign_pct, malignant_pct):
    """Determine risk level based on predictions"""
    if predicted_class == 'benign':
        if benign_pct > 90:
            return 'Very Low Risk'
        elif benign_pct > 75:
            return 'Low Risk'
        else:
            return 'Moderate - Requires Verification'
    else:
        if malignant_pct > 90:
            return 'High Risk - Immediate Attention Required'
        elif malignant_pct > 75:
            return 'Elevated Risk - Consult Specialist Soon'
        else:
            return 'Moderate Risk - Further Testing Needed'

def get_recommendations(predicted_class, confidence):
    """Generate medical recommendations"""
    if predicted_class == 'benign':
        return {
            'urgency': 'routine',
            'actions': [
                'Continue regular skin monitoring',
                'Schedule routine dermatology check-ups',
                'Practice sun protection measures',
                'Monitor for any changes in size, shape, or color'
            ]
        }
    else:
        return {
            'urgency': 'urgent',
            'actions': [
                'Consult a dermatologist immediately',
                'Request professional biopsy examination',
                'Do not delay medical consultation',
                'Bring these results to your appointment',
                'Consider getting a second opinion'
            ]
        }

@app.route('/')
def index():
    """Serve the HTML frontend"""
    if os.path.exists('index.html'):
        return send_file('index.html')
    else:
        return jsonify({
            'message': 'Skin Cancer Detection API',
            'version': '2.0',
            'status': 'running',
            'endpoints': {
                '/': 'Serves web interface',
                '/health': 'Health check (GET)',
                '/predict': 'Image prediction (POST)',
                '/stats': 'Model statistics (GET)'
            }
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        logger.info(f"Processing image: {image_file.filename}")
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Extract probabilities
        benign_prob = float(probabilities[0][0].item())
        malignant_prob = float(probabilities[0][1].item())
        
        benign_pct = benign_prob * 100
        malignant_pct = malignant_prob * 100
        
        predicted_class = CLASS_LABELS[predicted.item()]
        confidence_value = float(confidence.item())
        
        # Determine risk level
        risk_level = get_risk_level(predicted_class, benign_pct, malignant_pct)
        
        # Get recommendations
        recommendations = get_recommendations(predicted_class, confidence_value)
        
        # Prepare response
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'confidence': confidence_value,
            'risk_level': risk_level,
            'percentages': {
                'benign': benign_pct,
                'malignant': malignant_pct
            },
            'all_probabilities': {
                'benign': benign_prob,
                'malignant': malignant_prob
            },
            'interpretation': {
                'primary_finding': f"{predicted_class.capitalize()} tissue detected",
                'confidence_level': f"{confidence_value * 100:.1f}%",
                'benign_likelihood': f"{benign_pct:.1f}%",
                'malignant_likelihood': f"{malignant_pct:.1f}%"
            },
            'recommendations': recommendations,
            'disclaimer': 'This is a preliminary screening tool. Always consult a qualified healthcare professional for diagnosis.'
        }
        
        # Log prediction
        logger.info("="*60)
        logger.info("PREDICTION RESULT")
        logger.info(f"Class: {predicted_class.upper()}")
        logger.info(f"Confidence: {confidence_value * 100:.2f}%")
        logger.info(f"Benign: {benign_pct:.2f}% | Malignant: {malignant_pct:.2f}%")
        logger.info(f"Risk Level: {risk_level}")
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'num_classes': NUM_CLASSES,
        'class_labels': CLASS_LABELS,
        'model_type': 'Vision Transformer (ViT)',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    try:
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return jsonify({
            'model_info': {
                'architecture': 'vit_base_patch16_224',
                'total_parameters': param_count,
                'trainable_parameters': trainable_params,
                'classes': CLASS_LABELS,
                'input_size': '224x224',
                'device': str(device)
            },
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    print(f"🧠 Model: Vision Transformer (ViT)")
    print(f"📊 Classes: {', '.join(CLASS_LABELS)}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"📄 Interface: Open index.html in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)