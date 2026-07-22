
import cv2
import numpy as np
from pathlib import Path
from src.features.extractors import extract_research_fusion, extract_sift_features, extract_hog_features, extract_gabor_features
from src.preprocessing.processor import detect_and_align_face
import json

CUSTOM_FACES_DIR = Path("Faces")
# Load index
with open("Faces/index.json") as f:
    index = json.load(f)
    
for face_id, data in index["registered_faces"].items():
    print(f"\n--- Face: {data['name']} ---")
    print(f"Avg encoding len: {len(data['avg_encoding'])}")
    print(f"First 10 values: {data['avg_encoding'][:10]}")
    
# Let's test loading a test image (if any)
test_images = list(CUSTOM_FACES_DIR.glob("*/*.jpg"))
for img_path in test_images[:2]:
    print(f"\n--- Testing image: {img_path} ---")
    img = cv2.imread(str(img_path))
    aligned = detect_and_align_face(img)
    print(f"Aligned shape: {aligned.shape}")
    
    sift = extract_sift_features(aligned)
    hog = extract_hog_features(aligned)
    gabor = extract_gabor_features(aligned)
    fused = extract_research_fusion(aligned)
    
    print(f"SIFT len: {len(sift)}, first 5: {sift[:5]}")
    print(f"HOG len: {len(hog)}, first 5: {hog[:5]}")
    print(f"Gabor len: {len(gabor)}, first 5: {gabor[:5]}")
    print(f"Fused len: {len(fused)}")
    
# Now let's see what happens when comparing all!
if index["registered_faces"]:
    print("\n--- Comparing all registered faces with each other ---")
    registered = list(index["registered_faces"].values())
    for i in range(len(registered)):
        for j in range(len(registered)):
            if i != j:
                v1 = np.array(registered[i]["avg_encoding"])
                v2 = np.array(registered[j]["avg_encoding"])
                score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                print(f"{registered[i]['name']} <-> {registered[j]['name']}: {score:.4f} (confidence: {(score+1)/2*100:.2f}%)")
