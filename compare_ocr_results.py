import csv
import os

def compare_results():
    paddle_csv = "data/pipeline_output/pipeline_results.csv"
    tesseract_csv = "data/pipeline_output_hybrid/results_hybrid.csv"
    
    if os.path.exists(paddle_csv):
        with open(paddle_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            paddle_rows = list(reader)
            paddle_success = len([r for r in paddle_rows if r.get('name', '').strip()])
            print(f"PaddleOCR:  {paddle_success}/{len(paddle_rows)} names found")
    else:
        print("Paddle results not found.")

    if os.path.exists(tesseract_csv):
        with open(tesseract_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            tess_rows = list(reader)
            tess_success = len([r for r in tess_rows if r.get('name', '').strip()])
            print(f"Tesseract:  {tess_success}/{len(tess_rows)} names found")
            
            # Print intersection
            # Find names found by Tesseract but not Paddle
            paddle_names = {r['image_id']: r.get('name', '').strip() for r in paddle_rows}
            tess_names = {r['image_id']: r.get('name', '').strip() for r in tess_rows}
            
            print("\nImproved by Tesseract:")
            for img_id, name in tess_names.items():
                if name and not paddle_names.get(img_id):
                    print(f"  {img_id}: '{name}'")
                    
            print("\nLost by Tesseract:")
            for img_id, name in paddle_names.items():
                if name and not tess_names.get(img_id):
                    print(f"  {img_id}: '{name}' (Paddle saw '{name}', Tesseract empty)")

if __name__ == "__main__":
    compare_results()
