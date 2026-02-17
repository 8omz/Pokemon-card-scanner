import os
import json
import glob
import re
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.rotomcv.db_client import RotomDB

def clean_str(s):
    if not s: return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def clean_number(num_str):
    if not num_str: return ""
    parts = num_str.split('/')
    # If 015/086, return 15
    # If 15, return 15
    val = parts[0]
    if val.isdigit():
        return str(int(val))
    
    # If TG01, return TG01
    return val

def ingest():
    db = RotomDB().db
    cards_col = db["cards"]
    
    cards_dir = os.path.join(project_root, "cards", "en")
    files = glob.glob(os.path.join(cards_dir, "*.json"))
    
    print(f"Found {len(files)} set files in {cards_dir}")
    
    total_inserted = 0
    total_skipped = 0
    
    for f in files:
        set_id = os.path.basename(f).replace(".json", "")
        with open(f, 'r', encoding='utf-8') as fp:
            try:
                data = json.load(fp)
                for card in data:
                    # Prepare document
                    name = card.get('name', '')
                    number = card.get('number', '')
                    
                    if not number:
                        continue 

                    clean_n = clean_str(name)
                    clean_num = clean_number(number)
                    
                    doc = card.copy()
                    doc["set_id"] = set_id
                    doc["name_clean"] = clean_n
                    doc["number_clean"] = clean_num
                    
                    # Create unique ID to prevent duplicates
                    # set_id + number is usually unique, but sometimes variants exist?
                    # Let's use set_id + number as key for upsert
                    filter_query = {"set_id": set_id, "number": number}
                    
                    try:
                        cards_col.replace_one(filter_query, doc, upsert=True)
                        total_inserted += 1
                    except Exception as e:
                        print(f"Error inserting {name}: {e}")
                        total_skipped += 1
                        
            except Exception as e:
                print(f"Error reading {f}: {e}")

    print(f"Ingestion Complete.")
    print(f"Processed/Upserted: {total_inserted}")
    print(f"Total in DB: {cards_col.count_documents({})}")

if __name__ == "__main__":
    ingest()
