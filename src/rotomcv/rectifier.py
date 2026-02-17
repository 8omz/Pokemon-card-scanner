import cv2
import numpy as np
import os
import sys
import datetime


# Add directory to path to import annotator utils if needed
# Assuming this script runs from project root or src location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from annotator import load_manifest, project_root, dataset_csv_path
except ImportError:
    # Fallback if running relative to file
    from .annotator import load_manifest, project_root, dataset_csv_path

# Constants
TARGET_WIDTH = 600
TARGET_HEIGHT = 840
OUTPUT_DIR_PHONE = os.path.join(project_root, "data", "annotated", "phone")
RECTIFICATION_LOG_PATH = os.path.join(project_root, "data", "manifests", "rectification_log.csv")

def append_rectifier_log(image_id, status, reason=None):
    file_exists = os.path.exists(RECTIFICATION_LOG_PATH)
    need_header = False
    if not file_exists or (file_exists and os.path.getsize(RECTIFICATION_LOG_PATH)) == 0:
        need_header = True
    
    if need_header:
          with open(RECTIFICATION_LOG_PATH,"a") as file:
              file.write("TIMESTAMP,image_id,status,reason\n")
              
    with open(RECTIFICATION_LOG_PATH,"a") as file:
        current_time = datetime.datetime.now().isoformat()
        if reason:
            file.write(f"{current_time},{image_id},{status},{reason}\n")
        else:
            file.write(f"{current_time},{image_id},{status}\n")


def get_destination_points(width, height, padding=0):
    """
    Returns the 4 destination points for the perspective transform.
    Order: TL, TR, BR, BL
    """
    # If padding is used, we shift the destination points inside the larger canvas
    # But usually 'width' and 'height' are the desired output size of the *card*.
    # If we want margin, the output image size should be larger.
    
    # Standard output:
    # (0, 0) -> (w, 0) -> (w, h) -> (0, h)
    
    if padding == 0:
        return np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype="float32")
    else:
        # If we want padding AROUND the card, the destination image size needs to be larger.
        # But here we define where the card corners go.
        # If the output image is also WxH, then padding means the card is smaller.
        # If the output image is (W+2p)x(H+2p), then card points are offset by p.
        return np.array([
            [padding, padding],
            [width - padding, padding],
            [width - padding, height - padding],
            [padding, height - padding]
        ], dtype="float32")

def unwarp_card(image, src_corners, width=TARGET_WIDTH, height=TARGET_HEIGHT, padding=10):
    """
    Unwarps the card defined by src_corners from image.
    Returns the flattened image.
    """
    if len(src_corners) != 4:
        return None
        
    src_pts = np.array(src_corners, dtype="float32")
    
    # Destination points.
    # We want the output image to include the padding.
    # So effective output size = width + 2*padding, height + 2*padding
    out_w = width + 2 * padding
    out_h = height + 2 * padding
    
    # The card corners map to (padding, padding) ...
    dst_pts = np.array([
        [padding, padding],
        [out_w - padding, padding],
        [out_w - padding, out_h - padding],
        [padding, out_h - padding]
    ], dtype="float32")

    # Compute Homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warping
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    
    return warped

def enhance_card(image):
    """
    Applies basic enhancement (brightness/contrast).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced

def batch_process_manifest(manifest_path, output_dir):
    data = load_manifest(manifest_path)
    count = 0
    skipped = 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting batch processing of {len(data)} images...")
    
    for row in data:
        image_id = row['image_id']
        
        # Check if fully annotated
        if row['TL_x'] is None or row['BL_y'] is None:
            # print(f"Skipping {image_id} (not annotated)")
            skipped += 1
            # append_rectifier_log(image_id, "SKIPPED", "Not annotated") # Too verbose maybe? 
            # User wants similar logic. Annotator logs "ANNOTATED" vs errors.
            # Here we might want to log failures or successes.
            # I will skip logging for "not annotated" to keep it clean, or log it?
            # Let's log errors/success only for now to avoid spamming 100s of skipped images.
            continue

            
        # Get Corners
        try:
            corners = [
                (int(row['TL_x']), int(row['TL_y'])),
                (int(row['TR_x']), int(row['TR_y'])),
                (int(row['BR_x']), int(row['BR_y'])),
                (int(row['BL_x']), int(row['BL_y']))
            ]
        except (ValueError, TypeError):
             print(f"Skipping {image_id} (corrupt coordinates)")
             append_rectifier_log(image_id, "ERROR", "Corrupt coordinates")
             skipped += 1
             continue


        # Load Image
        img_path = row['image_path']
        image = cv2.imread(img_path)
        if image is None:
             print(f"Error reading {img_path}")
             append_rectifier_log(image_id, "ERROR", f"Could not read image: {img_path}")
             continue

             
        # Unwarp
        rect_image = unwarp_card(image, corners)
        
        if rect_image is not None:
            # Enhance
            # enhanced_image = enhance_card(rect_image) # Optional: Enable if desired
            final_image = rect_image
            
            # Save
            out_filename = f"{image_id}_rect.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, final_image)
            count += 1
            print(f"Processed: {out_filename}")
            append_rectifier_log(image_id, "RECTIFIED")

            
    print(f"Done! Processed {count} images. Skipped {skipped}.")

if __name__ == "__main__":
    batch_process_manifest(dataset_csv_path, OUTPUT_DIR_PHONE)
