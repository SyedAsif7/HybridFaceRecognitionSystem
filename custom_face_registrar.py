
import os
import json
import cv2
import numpy as np
from pathlib import Path
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_hybrid_features, extract_research_fusion

# Configuration
CUSTOM_FACES_DIR = Path("Faces")
CUSTOM_FACES_INDEX = Path("Faces/index.json")
ENCODINGS_FILE = Path("encodings/encodings.pkl")

def ensure_directories():
    """Ensure required directories exist"""
    CUSTOM_FACES_DIR.mkdir(exist_ok=True)
    ENCODINGS_FILE.parent.mkdir(exist_ok=True)

def load_face_index():
    """Load the face index database"""
    if not CUSTOM_FACES_INDEX.exists():
        return {"registered_faces": {}, "next_id": 1}
    with open(CUSTOM_FACES_INDEX, "r") as f:
        return json.load(f)

def save_face_index(index):
    """Save the face index to file"""
    with open(CUSTOM_FACES_INDEX, "w") as f:
        json.dump(index, f, indent=2)

def register_face(person_name, face_images_dir):
    """
    Register a new person with their face images
    
    Args:
        person_name: Name of the person
        face_images_dir: Directory containing their face images
    """
    ensure_directories()
    index = load_face_index()
    
    # Create person's directory in Faces
    person_dir = CUSTOM_FACES_DIR / person_name.replace(" ", "_")
    person_dir.mkdir(exist_ok=True)
    
    # Process images
    encodings = []
    source_dir = Path(face_images_dir)
    
    for img_path in source_dir.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".pgm"]:
            # Read and process image
            img = cv2.imread(str(img_path))
            if img is not None:
                # Detect and align face
                aligned_face = detect_and_align_face(img)
                
                # Save aligned face to person's directory
                output_path = person_dir / f"{len(encodings) + 1}.jpg"
                cv2.imwrite(str(output_path), (aligned_face * 255).astype(np.uint8))
                
                # Extract features for encoding
                features = extract_research_fusion(aligned_face)
                encodings.append(features)
    
    if len(encodings) == 0:
        print(f"No valid face images found for {person_name}")
        return False
    
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
    save_face_index(index)
    
    # Update class names
    update_class_names(index)
    
    print(f"Successfully registered {person_name} with {len(encodings)} images!")
    return True

def update_class_names(face_index):
    """Update the class names file to include registered faces"""
    class_names = []
    for face_id in sorted(face_index["registered_faces"].keys(), key=int):
        class_names.append(face_index["registered_faces"][face_id]["name"])
    
    class_data = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "dataset": "Custom Face Database",
        "feature_dim": 8317,
        "img_size": [128, 128]
    }
    
    with open("weights/class_names.json", "w") as f:
        json.dump(class_data, f, indent=2)

def get_registered_faces():
    """Get list of all registered faces"""
    index = load_face_index()
    return list(index["registered_faces"].values())

def recognize_face_with_custom(face_image, method="cosine"):
    """
    Recognize a face using custom registered faces
    
    Args:
        face_image: Aligned face image
        method: Recognition method (cosine, euclidean)
    
    Returns:
        Recognition result dict
    """
    index = load_face_index()
    if not index["registered_faces"]:
        return None
    
    # Extract features from query face
    query_features = extract_research_fusion(face_image)
    
    best_match = None
    best_score = -float('inf')
    
    for face_id, face_data in index["registered_faces"].items():
        stored_encoding = np.array(face_data["avg_encoding"])
        
        if method == "cosine":
            # Cosine similarity
            score = np.dot(query_features, stored_encoding) / (
                np.linalg.norm(query_features) * np.linalg.norm(stored_encoding)
            )
        else:  # euclidean
            # Negative Euclidean distance (higher is better)
            score = -np.linalg.norm(query_features - stored_encoding)
        
        if score > best_score:
            best_score = score
            best_match = face_data
    
    # Determine confidence level
    if method == "cosine":
        confidence = (best_score + 1) / 2 * 100  # convert to 0-100
    else:
        # Normalize euclidean distance to 0-100
        confidence = max(0, 100 - abs(best_score) * 10)
    
    return {
        "name": best_match["name"],
        "id": best_match["id"],
        "confidence": confidence,
        "score": float(best_score)
    }

if __name__ == "__main__":
    # Example usage
    print("Custom Face Registrar")
    print("=" * 50)
    
    # Show registered faces
    registered = get_registered_faces()
    if registered:
        print(f"Registered faces ({len(registered)}):")
        for face in registered:
            print(f"  - {face['name']} (ID: {face['id']}, {face['num_images']} images)")
    else:
        print("No faces registered yet.")

