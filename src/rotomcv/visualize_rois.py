"""
ROI Visualizer - Shows what regions we're OCR'ing

This tool creates annotated images showing the ROI boxes overlaid on cards
so we can see if we're capturing the right areas.
"""

import cv2
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roi_explorer import ROI_DEFINITIONS, get_roi_crops
from annotator import project_root

INPUT_DIR = os.path.join(project_root, "data", "annotated", "phone")
OUTPUT_DIR = os.path.join(project_root, "data", "roi_debug")

def visualize_rois(image_path, output_path):
    """
    Draw ROI boxes on the image and save it for visual inspection
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return
    
    # Create a copy for drawing
    annotated = image.copy()
    
    # Define colors for different ROIs
    colors = {
        "name_header": (0, 255, 0),      # Green
        "card_number": (255, 0, 0),      # Blue
        "set_icon": (0, 0, 255),         # Red
        "art_window": (255, 255, 0)      # Cyan
    }
    
    # Draw each ROI
    for roi_name, (x1, y1, x2, y2) in ROI_DEFINITIONS.items():
        color = colors.get(roi_name, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = roi_name.replace('_', ' ').title()
        cv2.putText(annotated, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save annotated image
    cv2.imwrite(output_path, annotated)
    print(f"Saved ROI visualization: {output_path}")
    
    # Also extract and save individual ROIs
    crops = get_roi_crops(image)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    for roi_name, crop in crops.items():
        crop_path = os.path.join(OUTPUT_DIR, f"{base_name}_{roi_name}.jpg")
        cv2.imwrite(crop_path, crop)
    
    return annotated

def process_samples(num_samples=5):
    """
    Process multiple sample images to see ROI placement
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get rectified images
    search_path = os.path.join(INPUT_DIR, "*_rect.jpg")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No rectified images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(files)} rectified images")
    print(f"Processing {min(num_samples, len(files))} samples...\n")
    
    # Process a variety of samples
    sample_indices = [0, 7, 14, 21, 28]  # Different cards
    
    for idx in sample_indices[:num_samples]:
        if idx >= len(files):
            break
            
        input_path = files[idx]
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
        
        print(f"Processing: {filename}")
        visualize_rois(input_path, output_path)
    
    print(f"\n✅ ROI visualizations saved to: {OUTPUT_DIR}")
    print(f"📝 Check the images to see if the ROI boxes are in the right positions!")

if __name__ == "__main__":
    process_samples(num_samples=10)
