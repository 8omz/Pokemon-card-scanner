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
    "name_header": (90, 30, 520, 80),      # OCR: Pokemon Name
    "card_number_bl": (15, 760, 260, 840), # OCR: Bottom Left - Expanded boundaries
    "card_number_br": (380, 760, 585, 840),# OCR: Bottom Right - Expanded boundaries
    "card_number_center": (180, 760, 420, 840), # OCR: Center Bottom - Expanded boundaries
    "card_bottom_full": (0, 740, 600, 840), # OCR: Full Bottom Scan for Paddle
    "art_window": (50, 100, 550, 450)      # Hash: Art Window
}

INPUT_DIR = os.path.join(project_root, "data", "annotated", "phone")
OUTPUT_DIR = os.path.join(project_root, "data", "roi_samples")

def get_roi_crops(image):
    """
    Extracts ROIs from the input image and returns them as a dictionary of numpy arrays.
    
    Args:
        image (numpy.ndarray): The input image (expected 600x840).
        
    Returns:
        dict: A dictionary mapping ROI names to cropped image arrays.
    """
    crops = {}
    if image is None:
        return crops
        
    h, w = image.shape[:2]
    
    for roi_name, (x1, y1, x2, y2) in ROI_DEFINITIONS.items():
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Crop
        roi_crop = image[y1:y2, x1:x2]
        crops[roi_name] = roi_crop
        
    return crops

def save_roi_crops(crops, output_dir, prefix):
    """
    Saves a dictionary of ROI crops to disk.
    
    Args:
        crops (dict): Dictionary of ROI names to image arrays.
        output_dir (str): Directory to save images.
        prefix (str): Prefix for filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for roi_name, crop in crops.items():
        filename = f"{prefix}_{roi_name}_crop.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, crop)
        print(f"Saved {roi_name} to {save_path}")

def extract_rois(image, output_dir, prefix):
    """
    Wrapper for backward compatibility. Extracts and saves ROIs.
    """
    crops = get_roi_crops(image)
    save_roi_crops(crops, output_dir, prefix)

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
        
    extract_rois(image, OUTPUT_DIR, image_id)
    print("ROI extraction complete.")

if __name__ == "__main__":
    process_sample()
