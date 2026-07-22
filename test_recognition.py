
"""Test script to verify face recognition works"""
import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_research_fusion

def load_custom_face_index():
    index_path = Path("Faces/index.json")
    if index_path.exists():
        with open(index_path, "r") as f:
            return json.load(f)
    return {"registered_faces": {}, "next_id": 1}

def recognize_custom_face(image):
    index = load_custom_face_index()
    if not index["registered_faces"]:
        print("No registered faces found!")
        return None
    
    # Detect and align the face
    aligned_face = detect_and_align_face(image)
    
    # Extract features
    query_features = extract_research_fusion(aligned_face)
    
    best_match = None
    best_score = -float('inf')
    
    print("\nComparing with registered faces:")
    for face_id, face_data in index["registered_faces"].items():
        stored_encoding = np.array(face_data["avg_encoding"])
        
        # Cosine similarity
        score = np.dot(query_features, stored_encoding) / (
            np.linalg.norm(query_features) * np.linalg.norm(stored_encoding)
        )
        
        print(f"  - {face_data['name']}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_match = face_data
    
    confidence = (best_score + 1) / 2 * 100
    
    print("\n" + "="*50)
    print("RECOGNITION RESULT:")
    print("="*50)
    print(f"Name: {best_match['name']}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Similarity Score: {best_score:.4f}")
    print("="*50)
    
    return {
        "name": best_match["name"],
        "confidence": confidence,
        "score": best_score
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_recognition.py <image_path>")
        return 1
    
    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not load image: {image_path}")
        return 1
    
    print("Testing face recognition...")
    recognize_custom_face(img)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
