import cv2
import os
import sys
import glob

# Add directory to path to import annotator utils if needed
# Assuming this script runs from project root or src location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from annotator import project_root
except ImportError:
    # Use fallback if import fails, though usually we run from root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
# Standard ROI definitions for 600x840 card
ROI_DEFINITIONS = {
    "name_header": (20, 30, 580, 80),      # OCR: Pokemon Name
    "set_icon": (480, 760, 580, 810),      # Template: Set Icon
    "card_number": (30, 800, 190, 840),    # OCR: Card Number (e.g. 015/198)
    "art_window": (50, 100, 550, 450)      # Hash: Art Window
}

INPUT_DIR = os.path.join(project_root, "data", "annotated", "phone")
OUTPUT_DIR = os.path.join(project_root, "data", "roi_samples")

def extract_rois(image):
    """
    Extracts ROIs from a 600x840 image.
    Returns a dictionary {feature_name: crop_image}
    """
    rois = {}
    h, w = image.shape[:2]
    
    # Sanity check dimensions if strictly required, or just proceed
    # Standard is 600x840.
    
    for feature, (x1, y1, x2, y2) in ROI_DEFINITIONS.items():
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        crop = image[y1:y2, x1:x2]
        rois[feature] = crop
        
    return rois

def save_rois(rois, output_dir, image_id):
    """
    Saves extracted ROIs to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for feature, crop in rois.items():
        filename = f"{image_id}_{feature}_crop.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, crop)
        print(f"Saved {feature} to {path}")

def process_sample():
    """
    Finds a sample image and processes it.
    """
    # Find first jpg in input dir
    search_path = os.path.join(INPUT_DIR, "*_rect.jpg")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No rectified images found in {INPUT_DIR}")
        return
        
    # Pick the first one
    sample_file = files[0]
    image_id = os.path.basename(sample_file).replace("_rect.jpg", "")
    
    print(f"Processing sample: {sample_file} ({image_id})")
    
    image = cv2.imread(sample_file)
    if image is None:
        print("Failed to load image.")
        return
        
    rois = extract_rois(image)
    save_rois(rois, OUTPUT_DIR, image_id)
    print("ROI extraction complete.")

if __name__ == "__main__":
    process_sample()
