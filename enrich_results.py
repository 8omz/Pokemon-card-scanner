import csv
import json
import os
from src.rotomcv.local_card_db import LocalCardDatabase

def enrich_data():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(project_root, "data", "pipeline_output_hybrid", "results_hybrid.csv")
    output_json = os.path.join(project_root, "data", "enriched_results.json")
    cards_dir = os.path.join(project_root, "cards") # Where user put the folders
    
    print(f"Initializing Local Database from {cards_dir}...")
    db = LocalCardDatabase(cards_dir)
    
    enriched_dict = {}
    
    print(f"Reading OCR results from {input_csv}...")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ocr_name = row.get('name', '').strip()
            ocr_number = row.get('number', '').strip()
            image_id = row.get('image_id', '')
            status = row.get('status', '')
            
            if not ocr_name:
                continue

            # Search DB
            matches = db.search_card(ocr_name, ocr_number)
            
            # Select best match
            selected_card = None
            match_type = "none"
            
            if matches:
                # If we have multiple, try to filter?
                # For now, take first. 
                # Ideally, if we have multiple "Charizard", we need number. 
                # If number was passed and matched, we are good.
                # If number failed or wasn't passed, we have a list of variants.
                
                # Check if we have an exact number match in the list
                exact_num_match = [m for m in matches if db._clean_number(m.get('number')) == db._clean_number(ocr_number)]
                
                if exact_num_match:
                    selected_card = exact_num_match[0]
                    match_type = "exact_number"
                else:
                    selected_card = matches[0] # Fallback to first
                    match_type = "name_only_best_guess"
            
            entry = {
                "scan_id": image_id,
                "ocr_name": ocr_name,
                "ocr_number": ocr_number,
                "status": status,
                "db_match": match_type,
                "card_data": selected_card
            }
            # Use dict to deduplicate by ID, keeping latest
            enriched_dict[image_id] = entry
            
    enriched_data = list(enriched_dict.values())
            
    print(f"Writing {len(enriched_data)} enriched records to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2)

if __name__ == "__main__":
    enrich_data()
