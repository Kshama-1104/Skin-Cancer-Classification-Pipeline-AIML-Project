from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms
import timm
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration - UPDATE THESE VALUES
NUM_CLASSES = 2  # Two classes: benign and malignant
CLASS_LABELS = ['benign', 'malignant']  # Your class names

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create the model architecture using timm
print("Loading model...")
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)

# Load the trained weights
try:
    # Load the state dictionary
    state_dict = torch.load('model.pt', map_location=device)
    
    # Handle different save formats
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            # Direct state_dict
            model.load_state_dict(state_dict)
    else:
        # If it's the complete model
        model = state_dict
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nPlease check:")
    print("1. Is 'model.pt' in the same directory as app.py?")
    print("2. Does NUM_CLASSES match your training configuration?")
    print("3. Was the model trained with timm library?")
    raise

# Define image preprocessing (adjust according to your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    """Serve the HTML frontend"""
    if os.path.exists('index.html'):
        return send_file('index.html')
    else:
        return jsonify({
            'message': 'ViT Image Classifier API',
            'endpoints': {
                '/': 'This page (serves index.html if available)',
                '/health': 'Health check (GET)',
                '/predict': 'Image prediction (POST)'
            },
            'note': 'Open index.html directly in your browser to use the web interface'
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Prepare response
        result = {
            'predicted_class': CLASS_LABELS[predicted.item()],
            'confidence': float(confidence.item()),
            'all_probabilities': {
                CLASS_LABELS[i]: float(probabilities[0][i].item()) 
                for i in range(len(CLASS_LABELS))
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True,
        'device': str(device),
        'num_classes': NUM_CLASSES
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ViT Model Server Starting...")
    print(f"Device: {device}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Server running on: http://localhost:5000")
    print(f"Open index.html in your browser to use the interface")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)