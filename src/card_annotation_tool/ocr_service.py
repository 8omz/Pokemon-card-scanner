import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
import cv2
import os
import sys
import re
import numpy as np
import glob

# Add directory to path if needed (though usually running as script or module)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize PaddleOCR
# use_angle_cls=True helps if cards are slightly rotated
# lang='en' uses the English model
# show_log=False reduces console spam
# Disable mkldnn to avoid windows internal error
# os.environ["FLAGS_use_mkldnn"] = "0" # Moved to top
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)






def preprocess_image(image):
    """
    Converts to grayscale and increases contrast to help OCR.
    """
    if image is None:
        return None
        
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase Contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # Alternative: Simple linear contrast
    # alpha = 1.5 # Contrast control (1.0-3.0)
    # beta = 0 # Brightness control (0-100)
    # contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Binarization (Optional, sometimes Paddle prefers raw gray or color)
    # ret, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return thresh
    
    # Bilateral Filter (Noise Reduction while keeping edges)
    # d=9, sigmaColor=75, sigmaSpace=75
    contrast = cv2.bilateralFilter(contrast, 9, 75, 75)
    
    # Add Padding (Breathing room for OCR)
    # 5px white border
    contrast = cv2.copyMakeBorder(contrast, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Convert back to BGR for PaddleOCR which expects 3 channels
    contrast_bgr = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    return contrast_bgr



def clean_card_number(text):
    """
    Extracts patterns like '015/198' using regex.
    """
    if not text:
        return ""
        
    # Look for digits/digits pattern
    match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    
    # Fallback: just return the text if no fraction found, or clean special chars
    # Maybe remove non-alphanumeric except /
    return text.strip()

def get_text_from_roi(roi_crop, is_number=False):
    """
    Runs OCR on a single crop.
    """
    if roi_crop is None:
        return ""
        
    # Preprocess
    processed_img = preprocess_image(roi_crop)
    
    # Run PaddleOCR
    # Optimize: det=False, cls=False to skip detection and angle class if crop is good.
    # Updated to use predict() to avoid deprecation warning.
    # det=False causing TypeError in this version, using standard predict.
    try:
        result = ocr.predict(processed_img)
    except TypeError:
         # Fallback just in case
         result = ocr.ocr(processed_img)


    # print(f"DEBUG Result: {result}", flush=True)
    # import sys; sys.exit(0)
    
    if result and isinstance(result, list) and len(result) > 0:
        res = result[0]
        
        # New Paddlex Structure (Dict)
        if isinstance(res, dict) and 'rec_texts' in res:
             texts = res['rec_texts']
             # You could filter by 'rec_scores' here if needed
             full_text = " ".join(texts)
             if is_number:
                 return clean_card_number(full_text)
             return full_text
             
        # Old PaddleOCR Structure (List of lists)
        if isinstance(res, list):
            texts = []
            for line in res:
                 # line structure: [text, confidence] when det=False
                 if len(line) >= 2:
                     text_content = line[0]
                     confidence = line[1]
                     
                     # Confidence Filtering
                     if confidence < 0.90:
                         print(f"Warning: Low confidence ({confidence:.2f}) for '{text_content}' - [FALLBACK TRIGGER]")
                         # You might want to return None or empty string to trigger fallback
                         # For now we just log it.
                         
                     texts.append(text_content)
            
            full_text = " ".join(texts)
            if is_number:
                return clean_card_number(full_text)
            return full_text

            
    return ""

def test_ocr_service():
    """
    Test OCR on the roi_samples directory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_dir = os.path.join(project_root, "data", "roi_samples")
    
    print(f"Testing OCR on samples in {samples_dir}...")
    
    # Find headers and numbers
    headers = glob.glob(os.path.join(samples_dir, "*_name_header_crop.jpg"))
    numbers = glob.glob(os.path.join(samples_dir, "*_card_number_crop.jpg"))
    
    print("\n--- Name Headers ---")
    for img_path in headers:
        img = cv2.imread(img_path)
        text = get_text_from_roi(img)
        print(f"{os.path.basename(img_path)} -> '{text}'")
        
    print("\n--- Card Numbers ---")
    for img_path in numbers:
        img = cv2.imread(img_path)
        text = get_text_from_roi(img, is_number=True)
        print(f"{os.path.basename(img_path)} -> '{text}'")

if __name__ == "__main__":
    test_ocr_service()
