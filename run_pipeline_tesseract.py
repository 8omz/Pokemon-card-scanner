import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.card_annotation_tool.pipeline_manager import CardPipeline

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_csv = os.path.join(project_root, "data", "manifests", "dataset.csv")
    
    # Use a separate output directory for clarity
    output_dir = os.path.join(project_root, "data", "pipeline_output_tesseract")
    
    print("Initializing Pipeline with Tesseract Engine...")
    start_init = time.time()
    pipeline = CardPipeline(output_dir=output_dir, ocr_engine_type="tesseract")
    print(f"Initialization took {time.time() - start_init:.2f}s")
    
    print("Running Batch Processing...")
    start_run = time.time()
    pipeline.run_batch(dataset_csv, output_csv_name="results_tesseract.csv")
    end_run = time.time()
    
    print(f"Total Run Time: {end_run - start_run:.2f}s")
