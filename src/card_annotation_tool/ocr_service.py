import os
import cv2
import numpy as np
import re
import time
import glob

# Optimize environment variables for speed
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

class PokemonCardOCR:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PokemonCardOCR, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the PaddleOCR engine with mobile-optimized settings.
        """
        print("Initializing PokemonCardOCR engine...")
        # Mobile Optimization Strategy:
        # 1. use_textline_orientation=False: We trust our rectifier; skipping angle check saves time.
        # 2. ocr_version='PP-OCRv4': Forces lightweight Mobile models.
        # 3. enable_mkldnn=False: Avoids Windows compatibility issues.
        self.ocr = PaddleOCR(
            use_textline_orientation=False, 
            lang='en', 
            enable_mkldnn=False, 
            ocr_version='PP-OCRv4'
        )

        self.warmup()

    def warmup(self):
        """
        Run a dummy inference to load models into memory.
        """
        print("Warming up OCR engine...")
        dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
        try:
            self.ocr.predict(dummy_img)
            print("OCR engine ready.")
        except Exception as e:
            print(f"Warmup warning: {e}")

    def preprocess_image(self, image):
        """
        Applies bilateral filtering and padding for optimal OCR accuracy.
        """
        if image is None:
            return None
            
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Bilateral Filter (Noise Reduction while keeping edges)
        contrast = cv2.bilateralFilter(contrast, 9, 75, 75)
        
        # Add Padding (Breathing room for OCR)
        contrast = cv2.copyMakeBorder(contrast, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Convert back to BGR
        contrast_bgr = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
        return contrast_bgr

    def clean_card_number(self, text):
        if not text:
            return ""
        match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return text.strip()

    def extract_text(self, roi_crop, is_number=False):
        """
        Main API method to get text from a cropped ROI.
        """
        if roi_crop is None:
            return ""

        processed_img = self.preprocess_image(roi_crop)
        
        try:
            # Predict
            result = self.ocr.predict(processed_img)
            
            # Parse result
            if result and isinstance(result, list) and len(result) > 0:
                res = result[0]
                
                # Handle Paddlex Dict Format
                if isinstance(res, dict) and 'rec_texts' in res:
                     texts = res['rec_texts']
                     full_text = " ".join(texts)
                     if is_number:
                         return self.clean_card_number(full_text)
                     return full_text
                     
                # Handle Standard List Format
                if isinstance(res, list):
                    texts = []
                    for line in res:
                         if len(line) >= 2:
                             text_content = line[0]
                             confidence = line[1]
                             
                             if confidence < 0.90:
                                 # Log warning but return what we have (or implement fallback logic here)
                                 # print(f"Low confidence: {text_content} ({confidence:.2f})")
                                 pass
                                 
                             texts.append(text_content)
                    
                    full_text = " ".join(texts)
                    if is_number:
                        return self.clean_card_number(full_text)
                    return full_text
                    
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
            
        return ""

def test_ocr_service():
    """
    Test harness for the PokemonCardOCR class.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_dir = os.path.join(project_root, "data", "roi_samples")
    
    print(f"Testing OCR on samples in {samples_dir}...")
    
    # Initialize Service
    service = PokemonCardOCR()
    
    headers = glob.glob(os.path.join(samples_dir, "*_name_header_crop.jpg"))
    numbers = glob.glob(os.path.join(samples_dir, "*_card_number_crop.jpg"))
    
    start_time = time.time()
    count = 0
    
    print("\n--- Name Headers ---")
    for img_path in headers:
        img = cv2.imread(img_path)
        text = service.extract_text(img)
        print(f"{os.path.basename(img_path)} -> '{text}'")
        count += 1
        
    print("\n--- Card Numbers ---")
    for img_path in numbers:
        img = cv2.imread(img_path)
        text = service.extract_text(img, is_number=True)
        print(f"{os.path.basename(img_path)} -> '{text}'")
        count += 1
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = (total_time / count) * 1000 if count > 0 else 0
    
    print(f"\nCompleted in {total_time:.4f} seconds.")
    print(f"Processed {count} ROIs.")
    print(f"Average time per ROI: {avg_time:.2f} ms")

if __name__ == "__main__":
    test_ocr_service()
