
import cv2
import numpy as np
from pathlib import Path
from src.features.extractors import extract_research_fusion
from src.preprocessing.processor import detect_and_align_face
import json

CUSTOM_FACES_DIR = Path("Faces")

# Let's create a dummy image or use one of the old ones!
# First check if we have any images in Faces/
test_images_dir = Path("Faces") / "test_images"
if not test_images_dir.exists():
    test_images_dir.mkdir(parents=True)
    # Create a dummy face image
    dummy_face = np.random.randint(0, 255, (128,128,3), dtype=np.uint8)
    cv2.imwrite(str(test_images_dir / "dummy_face.jpg"), dummy_face)

# Now let's see what happens when we register a person, then try to recognize a dummy!
print("--- Testing Not Found Logic ---")

# Register a test person with an existing image (if we have any old images)
old_faces_dir = Path("Faces_old")
if old_faces_dir.exists():
    print("Faces_old exists!")
else:
    print("No old faces, let's use some dummy test!")

# Let's also test with the current (empty) index
print("\nCurrent index is empty, so all should be Not Found!")
