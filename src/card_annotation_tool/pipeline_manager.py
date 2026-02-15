import os
import cv2
import csv
import glob
import time
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rectifier import unwarp_card
from roi_explorer import get_roi_crops
from ocr_service import PokemonCardOCR
try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

class CardPipeline:
    def __init__(self, output_dir="data/pipeline_output", ocr_engine_type="hybrid"):
        self.output_dir = output_dir
        self.ocr_engine_type = ocr_engine_type
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize OCR Engine (Warmup happens here)
        self.ocr_engine = PokemonCardOCR()
        
    def process_image(self, image_path, corners=None):
        """
        Runs the full pipeline on a single image.
        
        Args:
            image_path (str): Path to the input image.
            corners (list): List of 4 (x,y) tuples. If None, rectification is skipped/assumed or failed.
            
        Returns:
            dict: Result containing {filename, name, number, confidence, status, etc.}
        """
        filename = os.path.basename(image_path)
        result = {
            "filename": filename,
            "status": "pending",
            "name": "",
            "number": "",
            "confidence_flag": False
        }
        
        try:
            # 1. Load Image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # 2. Rectify
            if corners:
                rectified_img = unwarp_card(image, np.array(corners, dtype=np.float32))
            else:
                # If no corners provided, assume image is already rectified (e.g. from debug folder)
                # Or skip. For this pipeline, we will assume we have corners from the log.
                # If we are processing raw images without corners, we would need a detector here.
                # For now, let's assume valid corners are passed.
                rectified_img = image # CAUTION: Only if already rect.
            
            if rectified_img is None:
                 raise ValueError("Rectification failed")

            # 3. Extract ROIs (In-Memory)
            crops = get_roi_crops(rectified_img)
            
            # 4. OCR
            # Name
            name_text = self.ocr_engine.extract_text(crops.get('name_header'), engine=self.ocr_engine_type)
            
            # Number
            # Use new robust extraction method
            number_text = self.ocr_engine.extract_text_for_number(crops, engine=self.ocr_engine_type)
            
            # 5. Populate Result
            result["name"] = name_text
            result["number"] = number_text
            result["status"] = "success"
            
            # Simple Confidence Check (Placeholder logic)
            # Real confidence usually comes from the OCR engine, but we abstracted it.
            # We can flag if name is empty or number pattern looks wrong.
            if not name_text or len(name_text) < 3:
                result["confidence_flag"] = True
                
        except Exception as e:
            result["status"] = f"error: {str(e)}"
            result["confidence_flag"] = True
            
        return result

    def run_batch(self, dataset_csv_path, output_csv_name="pipeline_results.csv"):
        """
        Runs the pipeline on a batch of images defined in dataset.csv.
        Atomic writing to CSV.
        """
        output_csv_path = os.path.join(self.output_dir, output_csv_name)
        
        if not os.path.exists(dataset_csv_path):
            print(f"Error: Dataset not found at {dataset_csv_path}")
            return

        # Prepare Output CSV
        fieldnames = ["filename", "image_id", "status", "name", "number", "confidence_flag"]
        
        # Write header if new file
        write_header = not os.path.exists(output_csv_path)
        
        # Read dataset first to get total count for tqdm
        rows = []
        with open(dataset_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        print(f"Found {len(rows)} entries in dataset.")
        
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                
            # Process Loop
            for row in tqdm(rows, desc="Batch Processing"):
                image_id = row['image_id']
                raw_path = row['image_path']
                
                # Normalize path for Windows
                image_path = os.path.normpath(raw_path)
                
                # Get corners
                corners = None
                try:
                    # Check if corners exist in row and are valid numbers
                    if all(k in row and row[k] for k in ['corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y', 'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y']):
                        corners = [
                            [float(row['corner_tl_x']), float(row['corner_tl_y'])],
                            [float(row['corner_tr_x']), float(row['corner_tr_y'])],
                            [float(row['corner_br_x']), float(row['corner_br_y'])],
                            [float(row['corner_bl_x']), float(row['corner_bl_y'])]
                        ]
                except (ValueError, KeyError):
                    corners = None
                
                # Auto-Detection Fallback
                if corners is None:
                    # Initialize detector on demand or in __init__?
                    # Better to do it once. Let's assume self.detector exists.
                    if not hasattr(self, 'detector'):
                        from detector import CardDetector
                        self.detector = CardDetector()
                    
                    if os.path.exists(image_path):
                         # We need to read the image here to detect
                         # process_image reads it again. 
                         # Optimization: read once.
                         # But process_image interface takes path.
                         # Let's read, detect, pass corners.
                         temp_img = cv2.imread(image_path)
                         if temp_img is not None:
                             detected_corners, _ = self.detector.detect_card(temp_img)
                             if detected_corners is not None:
                                 corners = detected_corners.tolist()
                                 # print(f"  Auto-detected corners for {image_id}")
                    
                if corners is None:
                    print(f"Skipping {image_id}: No corners and auto-detection failed")
                    continue
                
                # Run Pipeline
                if not os.path.exists(image_path):
                     result = {
                        "filename": os.path.basename(image_path),
                        "image_id": image_id,
                        "status": "skipped_file_not_found",
                        "name": "",
                        "number": "",
                        "confidence_flag": True
                     }
                else:
                    result = self.process_image(image_path, corners)
                    result["image_id"] = image_id # Add ID to result
                
                # Atomic Write
                writer.writerow(result)
                f.flush()

if __name__ == "__main__":
    # Test Block
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_csv = os.path.join(project_root, "data", "manifests", "dataset.csv")
    
    pipeline = CardPipeline()
    pipeline.run_batch(dataset_csv)

