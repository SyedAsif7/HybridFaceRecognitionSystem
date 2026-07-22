
"""
Script to register a new face to the custom database
Usage: python register_face.py <person_name> <image_path1> <image_path2> ...
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import extract_research_fusion

CUSTOM_FACES_DIR = Path("Faces")
CUSTOM_FACES_INDEX = Path("Faces/index.json")

def load_face_index():
    if not CUSTOM_FACES_INDEX.exists():
        return {"registered_faces": {}, "next_id": 1}
    with open(CUSTOM_FACES_INDEX, "r") as f:
        return json.load(f)

def save_face_index(index):
    with open(CUSTOM_FACES_INDEX, "w") as f:
        json.dump(index, f, indent=2)

def update_class_names(face_index):
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
    
    print(f"Updated class names to: {class_names}")

def register_face(person_name, image_paths):
    ensure_dirs()
    index = load_face_index()
    
    # Create person directory
    person_dir = CUSTOM_FACES_DIR / person_name.replace(" ", "_")
    person_dir.mkdir(exist_ok=True)
    
    encodings = []
    saved_count = 0
    
    for img_path in image_paths:
        img_path_obj = Path(img_path)
        if img_path_obj.exists():
            try:
                img = cv2.imread(str(img_path_obj))
                if img is not None:
                    aligned_face = detect_and_align_face(img)
                    if aligned_face is not None:
                        # Save the aligned face
                        output_filename = f"{saved_count + 1}.jpg"
                        output_path = person_dir / output_filename
                        cv2.imwrite(str(output_path), (aligned_face * 255).astype(np.uint8))
                        
                        # Extract features
                        features = extract_research_fusion(aligned_face)
                        encodings.append(features)
                        saved_count += 1
                        print(f"✓ Processed: {img_path_obj.name}")
            except Exception as e:
                print(f"✗ Failed to process {img_path_obj.name}: {str(e)}")
    
    if saved_count == 0:
        print("\n⚠ No valid faces detected in the provided images!")
        return False
    
    # Update the index
    person_id = index["next_id"]
    index["registered_faces"][str(person_id)] = {
        "id": person_id,
        "name": person_name,
        "num_images": saved_count,
        "directory": str(person_dir.relative_to(CUSTOM_FACES_DIR.parent)),
        "avg_encoding": np.mean(encodings, axis=0).tolist()
    }
    index["next_id"] += 1
    save_face_index(index)
    
    update_class_names(index)
    
    print(f"\n✅ Successfully registered {person_name} with {saved_count} images!")
    return True

def ensure_dirs():
    CUSTOM_FACES_DIR.mkdir(exist_ok=True)
    Path("weights").mkdir(exist_ok=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: python register_face.py <person_name> <image_path1> <image_path2> ...")
        print("\nExample:")
        print("  python register_face.py \"Salman Khan\" salman1.jpg salman2.jpg")
        return 1
    
    person_name = sys.argv[1]
    image_paths = sys.argv[2:]
    
    print("=" * 60)
    print(f"Registering: {person_name}")
    print(f"Number of images: {len(image_paths)}")
    print("=" * 60)
    print()
    
    success = register_face(person_name, image_paths)
    
    if success:
        print("\n✅ Now you can upload an image of " + person_name + " and the system will recognize them!")
    else:
        print("\n❌ Registration failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
