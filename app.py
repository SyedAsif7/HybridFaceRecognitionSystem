"""
Flask Application - Hybrid Face Recognition System
====================================================
Author : Panchalwar Mam's Research - Enhanced Implementation
Features: SIFT|HOG|Gabor|Canny | Custom CNN | MobileNetV2 | SVM Comparison
          Eye-Landmark Alignment | Softmax/Sigmoid Switching | Confidence Scoring
          Custom Face Registration and Recognition
"""

import os
os.environ['KERAS_HOME'] = os.path.join(os.path.dirname(__file__), '.keras')
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2, json, base64, numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import (extract_hybrid_features, extract_research_fusion,
                                      extract_by_method)
from src.classifiers.svm_classifier import SVMClassifier
from src.models.hybrid_model import build_hybrid_model, build_mobilenetv2_model

# Custom face recognition imports
CUSTOM_FACES_DIR = Path("Faces")
CUSTOM_FACES_INDEX = Path("Faces/index.json")
last_rt_result = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

HISTORY_FILE = 'data/recognition_history.json'

# ── Load models ──────────────────────────────────────────────
model = model_mv2 = None
svm_classifiers = {}
class_names = []

# Load class names first
try:
    if os.path.exists('weights/class_names.json'):
        with open('weights/class_names.json', 'r') as f:
            class_names = json.load(f)['class_names']
        print(f"Loaded {len(class_names)} class names")
    else:
        class_names = [f"Person {i+1}" for i in range(40)]
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = [f"Person {i+1}" for i in range(40)]

# Build models from scratch and load weights
try:
    print("Building hybrid model...")
    model = build_hybrid_model()
    print("Hybrid model built successfully")
    if os.path.exists('weights/best_hybrid_model_weights.npz'):
        print("Loading hybrid model weights...")
        from src.models.hybrid_model import load_hybrid_model_weights
        model = load_hybrid_model_weights(model, 'weights/best_hybrid_model_weights.npz')
        print("Hybrid model weights loaded successfully")
except Exception as e:
    print(f"Error building/loading hybrid model: {e}")

try:
    print("Building MobileNetV2 model...")
    model_mv2 = build_mobilenetv2_model()
    print("MobileNetV2 model built successfully")
    # MobileNetV2 weights loading can be added later if needed
except Exception as e:
    print(f"Error building MobileNetV2 model: {e}")

for kernel in ['rbf', 'linear']:
    p = f'weights/svm_{kernel}.pkl'
    if os.path.exists(p):
        try:
            svm = SVMClassifier(kernel=kernel)
            svm.load(p)
            svm_classifiers[kernel] = svm
            print(f"SVM ({kernel}) loaded")
        except Exception as e:
            print(f"Error loading SVM ({kernel}): {str(e)[:100]}")

for d in ['data', app.config['UPLOAD_FOLDER'], 'weights', 'results']:
    os.makedirs(d, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)

# ── Custom Face Recognition ───────────────────────────────────────
def load_custom_face_index():
    """Load the custom face index database"""
    if not CUSTOM_FACES_INDEX.exists():
        return {"registered_faces": {}, "next_id": 1}
    with open(CUSTOM_FACES_INDEX, "r") as f:
        return json.load(f)

def recognize_custom_face(face_image):
    """Recognize face using custom registered faces"""
    index = load_custom_face_index()
    if not index["registered_faces"]:
        return None
    
    # Extract features
    query_features = extract_research_fusion(face_image)
    
    best_match = None
    best_score = -float('inf')
    
    for face_id, face_data in index["registered_faces"].items():
        stored_encoding = np.array(face_data["avg_encoding"])
        
        # Cosine similarity
        score = np.dot(query_features, stored_encoding) / (
            np.linalg.norm(query_features) * np.linalg.norm(stored_encoding)
        )
        
        if score > best_score:
            best_score = score
            best_match = face_data
    
    confidence = (best_score + 1) / 2 * 100  # convert to 0-100
    
    # Only return a match if confidence is above threshold (65%)
    if confidence < 65:
        return None
    
    return {
        "name": best_match["name"],
        "id": best_match["id"],
        "confidence": confidence,
        "score": float(best_score),
        "class_id": int(best_match["id"])
    }

def save_face_to_custom_db(person_name, face_images):
    """Save face to custom database"""
    index = load_custom_face_index()
    
    # Create person directory
    person_dir = CUSTOM_FACES_DIR / person_name.replace(" ", "_")
    person_dir.mkdir(exist_ok=True)
    
    encodings = []
    for idx, img in enumerate(face_images):
        aligned_face = detect_and_align_face(img)
        if aligned_face is not None:
            # Save image
            output_path = person_dir / f"{idx + 1}.jpg"
            cv2.imwrite(str(output_path), (aligned_face * 255).astype(np.uint8))
            
            # Extract features
            features = extract_research_fusion(aligned_face)
            encodings.append(features)
    
    if len(encodings) == 0:
        return False, "No valid faces detected in images"
    
    # Update index
    person_id = index["next_id"]
    index["registered_faces"][str(person_id)] = {
        "id": person_id,
        "name": person_name,
        "num_images": len(encodings),
        "directory": str(person_dir.relative_to(CUSTOM_FACES_DIR.parent)),
        "avg_encoding": np.mean(encodings, axis=0).tolist()
    }
    index["next_id"] += 1
    
    with open(CUSTOM_FACES_INDEX, "w") as f:
        json.dump(index, f, indent=2)
    
    # Update class names
    update_class_names(index)
    
    return True, f"Successfully registered {person_name} with {len(encodings)} images"

def update_class_names(face_index):
    """Update class names from custom face index"""
    global class_names
    new_class_names = []
    for face_id in sorted(face_index["registered_faces"].keys(), key=int):
        new_class_names.append(face_index["registered_faces"][face_id]["name"])
    
    class_data = {
        "class_names": new_class_names,
        "num_classes": len(new_class_names),
        "dataset": "Custom Face Database",
        "feature_dim": 8317,
        "img_size": [128, 128]
    }
    
    with open("weights/class_names.json", "w") as f:
        json.dump(class_data, f, indent=2)
    
    # Refresh the global class_names variable
    class_names = new_class_names

# ── Helpers ──────────────────────────────────────────────────
def confidence_level(conf_float):
    if conf_float >= 0.80: return "High"
    if conf_float >= 0.50: return "Medium"
    return "Low"

def save_history(record):
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        h.insert(0, record)
        with open(HISTORY_FILE, 'w') as f: json.dump(h[:500], f, indent=2)
    except: pass

def preprocess_for_model(face, use_mv2=False):
    if use_mv2:
        rgb = cv2.cvtColor((face*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return cv2.resize(rgb,(96,96)).reshape(1,96,96,3) / 255.0
    return face.reshape(1,128,128,1)

def run_prediction(img, method='hybrid', classifier='custom',
                   activation='softmax', fusion='research'):
    """
    method     : sift | hog | gabor | canny | hybrid | research_fusion
    classifier : custom | mobilenetv2 | svm_rbf | svm_linear | custom_faces
    activation : softmax | sigmoid
    fusion     : research (SIFT+HOG+Gabor) | full (SIFT+HOG+Gabor+Canny)
    """
    face = detect_and_align_face(img)

    # ── First, try Custom Face Recognition ───────────────
    index = load_custom_face_index()
    if index["registered_faces"]:
        custom_result = recognize_custom_face(face)
        if custom_result:
            conf_f = custom_result["confidence"] / 100
            return {
                'class_id':         custom_result["class_id"],
                'label':            custom_result["name"],
                'confidence':       f"{custom_result['confidence']:.2f}%",
                'confidence_level': confidence_level(conf_f),
                'top_predictions':  [{'class_id': custom_result["class_id"], 
                                      'label': custom_result["name"],
                                      'confidence': f"{custom_result['confidence']:.2f}%"}],
                'feature_count':    8317,
                'method':           'custom_faces',
                'fusion_type':      'research',
                'activation':       'N/A',
                'model_used':       'Custom Face Database (Hybrid Features)',
                'feature_breakdown': {'SIFT':'128-dim','HOG':'~8100-dim',
                                      'Gabor':'24-dim','Canny':'65-dim'},
            }
        else:
            # No confident match found
            return {
                'class_id':         -1,
                'label':            'Not Found',
                'confidence':       '0%',
                'confidence_level': 'low',
                'top_predictions':  [],
                'feature_count':    8317,
                'method':           'custom_faces',
                'fusion_type':      'research',
                'activation':       'N/A',
                'model_used':       'Custom Face Database (Hybrid Features)',
                'feature_breakdown': {'SIFT':'128-dim','HOG':'~8100-dim',
                                      'Gabor':'24-dim','Canny':'65-dim'},
            }

    # Feature extraction
    if method == 'hybrid':
        features = extract_hybrid_features(face)
    else:
        features = extract_by_method(face, method)

    # SVM branch
    # ── SVM branch ───────────────────────────────────────
    if classifier.startswith('svm'):
        kernel = classifier.split('_')[1] if '_' in classifier else 'rbf'
        if kernel not in svm_classifiers:
            raise RuntimeError(f"SVM ({kernel}) not trained yet. Train the model first.")
        return {**svm_classifiers[kernel].predict_single(features),
                'feature_count': int(len(features)),
                'method': method, 'fusion_type': fusion,
                'activation': 'N/A (SVM)'}

    # ── CNN / MobileNetV2 branch ──────────────────────────
    use_mv2 = (classifier == 'mobilenetv2' and model_mv2 is not None)
    active  = model_mv2 if use_mv2 else model
    if active is None:
        raise RuntimeError("No trained model found. Please train first.")

    img_input  = preprocess_for_model(face, use_mv2)
    feat_input = features.reshape(1, -1)

    raw = active.predict(
        {'image_input': img_input, 'feature_input': feat_input}, verbose=0
    )
    logits = raw[0] if isinstance(raw, list) else raw
    logits = logits[0]

    # ── Softmax / Sigmoid switching ───────────────────────
    if activation == 'sigmoid':
        probs = 1.0 / (1.0 + np.exp(-logits))   # element-wise sigmoid
        probs = probs / probs.sum()               # normalise to sum=1
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()                       # stable softmax

    top_idx = np.argsort(probs)[-5:][::-1]
    conf_f  = float(probs[top_idx[0]])

    # Get label from class_names (fallback if index out of bounds)
    def get_label(idx):
        if idx >=0 and idx < len(class_names):
            return class_names[idx]
        return f"Person {idx+1}"

    return {
        'class_id':         int(top_idx[0]),
        'label':            get_label(top_idx[0]),
        'confidence':       f"{conf_f*100:.2f}%",
        'confidence_level': confidence_level(conf_f),
        'top_predictions':  [{'class_id': int(i), 'label': get_label(i),
                              'confidence': f"{float(probs[i])*100:.2f}%"} for i in top_idx],
        'feature_count':    int(len(features)),
        'method':           method,
        'fusion_type':      fusion,
        'activation':       activation,
        'model_used':       'MobileNetV2' if use_mv2 else 'Custom CNN',
        'feature_breakdown': {'SIFT':'128-dim','HOG':'~8100-dim',
                              'Gabor':'24-dim','Canny':'65-dim'},
    }

# Video camera class for real-time feed
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.last_result = None
        self.frame_skip = 0
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            # Return dummy frame if camera fails
            ret, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            return jpeg.tobytes()
            
        # Draw rectangle on face (simple detection)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Run recognition every 3rd frame to avoid lag
        self.frame_skip += 1
        if self.frame_skip % 3 == 0 and len(faces) > 0:
            try:
                result = run_prediction(image)
                self.last_result = result
                global last_rt_result  # Need this to modify the global variable!
                last_rt_result = result
            except Exception as e:
                pass
        
        # Draw boxes and labels on detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle
            if self.last_result and self.last_result['label'] != 'Not Found':
                box_color = (0, 255, 0)  # Green for recognized face
            else:
                box_color = (0, 0, 255)  # Red for unknown face
                
            cv2.rectangle(image, (x, y), (x+w, y+h), box_color, 2)
            
            # Draw label if we have a result
            if self.last_result:
                label_text = f"{self.last_result['label']} ({self.last_result['confidence']})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size for background box
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), box_color, -1)
                
                # Draw label text
                cv2.putText(image, label_text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
                
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        model_loaded=(model is not None),
        mv2_loaded=(model_mv2 is not None),
        svm_loaded=bool(svm_classifiers))

@app.route('/predict', methods=['POST'])
def predict():
    if not request.files.get('file'):
        return jsonify({'error': 'No file uploaded'}), 400
    file       = request.files['file']
    method     = request.form.get('method', 'hybrid')
    classifier = request.form.get('classifier', 'custom')
    activation = request.form.get('activation', 'softmax')
    fusion     = request.form.get('fusion', 'research')
    filename   = secure_filename(file.filename)
    filepath   = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        img    = cv2.imread(filepath)
        result = run_prediction(img, method, classifier, activation, fusion)
        save_history({**result, 'timestamp': datetime.now().isoformat(), 'source': 'upload'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera-capture', methods=['POST'])
def camera_capture():
    try:
        data       = request.get_json()
        img_data   = data.get('image','').split(',')[-1]
        method     = data.get('method','hybrid')
        classifier = data.get('classifier','custom')
        activation = data.get('activation','softmax')
        fusion     = data.get('fusion','research')
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error':'Invalid image'}), 400
        result = run_prediction(img, method, classifier, activation, fusion)
        save_history({**result, 'timestamp': datetime.now().isoformat(), 'source': 'camera'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    files      = request.files.getlist('files[]')
    method     = request.form.get('method','hybrid')
    classifier = request.form.get('classifier','custom')
    activation = request.form.get('activation','softmax')
    fusion     = request.form.get('fusion','research')
    results, errors = [], []
    for file in files:
        try:
            fn = secure_filename(file.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            file.save(fp)
            res = run_prediction(cv2.imread(fp), method, classifier, activation, fusion)
            results.append({**res, 'filename': fn})
        except Exception as e:
            errors.append({'filename': file.filename, 'error': str(e)})
    return jsonify({'results':results,'errors':errors,
                    'success':len(results),'failed':len(errors)})

@app.route('/api/history')
def get_history():
    page = request.args.get('page',1,int)
    limit= request.args.get('limit',20,int)
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        s = (page-1)*limit
        return jsonify({'records':h[s:s+limit],'total':len(h),'page':page})
    except:
        return jsonify({'records':[],'total':0,'page':page})

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    with open(HISTORY_FILE,'w') as f: json.dump([],f)
    return jsonify({'message':'History cleared'})

@app.route('/api/rt-result')
def get_rt_result():
    return jsonify({'result': last_rt_result})

@app.route('/api/stats')
def get_stats():
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        confs = [float(r['confidence'].replace('%','')) for r in h if 'confidence' in r]
        avg   = sum(confs)/len(confs) if confs else 0
        return jsonify({
            'totalRecognitions': len(h),
            'averageConfidence': f"{avg:.2f}%",
            'personsCount': len(set(r.get('class_id',0) for r in h)),
            'modelLoaded': model is not None,
            'mv2Loaded':   model_mv2 is not None,
            'svmLoaded':   bool(svm_classifiers),
        })
    except Exception as e:
        return jsonify({'totalRecognitions':0,'averageConfidence':'0%',
                        'personsCount':0,'modelLoaded':False,'mv2Loaded':False,'svmLoaded':False})

@app.route('/api/comparison-results')
def comparison_results():
    return jsonify({
        'paper_results': [
            {'method':'SIFT + CNN','optimizer':'Adam','activation':'Softmax',
             'orl_acc':'92.50%','sheffield_acc':'100.00%','best':True},
            {'method':'HOG + CNN','optimizer':'Adamax','activation':'Softmax',
             'orl_acc':'91.25%','sheffield_acc':'97.50%','best':False},
            {'method':'Gabor + CNN','optimizer':'Adam','activation':'Softmax',
             'orl_acc':'93.75%','sheffield_acc':'95.00%','best':False},
            {'method':'Canny + CNN','optimizer':'SGD','activation':'Sigmoid',
             'orl_acc':'80.00%','sheffield_acc':'82.50%','best':False},
        ],
        'proposed': [
            {'method':'Research Fusion (SIFT+HOG+Gabor+CNN)','status':'Under Evaluation'},
            {'method':'Full Hybrid (SIFT+HOG+Gabor+Canny+CNN)','status':'Under Evaluation'},
            {'method':'MobileNetV2 Transfer Learning','status':'Under Evaluation'},
            {'method':'SVM (RBF kernel)','status':'Under Evaluation'},
            {'method':'SVM (Linear kernel)','status':'Under Evaluation'},
        ]
    })

# Custom face registration routes
@app.route('/api/register-face', methods=['POST'])
def register_face():
    try:
        person_name = request.form.get('name', '').strip()
        if not person_name:
            return jsonify({'error': 'Person name is required'}), 400
        
        face_images = []
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'At least one face image is required'}), 400
        
        for file in files:
            if file.filename:
                # Read the image
                nparr = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    face_images.append(img)
        
        success, message = save_face_to_custom_db(person_name, face_images)
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/registered-faces', methods=['GET'])
def get_registered_faces():
    index = load_custom_face_index()
    faces = list(index["registered_faces"].values())
    return jsonify({
        'faces': faces,
        'count': len(faces)
    })

# Real-time video feed route
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
