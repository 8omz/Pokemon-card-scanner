import json
import os
import glob
import re

class LocalCardDatabase:
    def __init__(self, cards_dir):
        self.cards_dir = cards_dir
        self.index_name_number = {} # (name_clean, number_clean) -> [card_data]
        self.index_name_only = {}   # name_clean -> [card_data]
        self._load_database()

    def _clean_str(self, s):
        if not s: return ""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _clean_number(self, num_str):
        # Handle "015/086" -> "15"
        # Handle "15" -> "15"
        if not num_str: return ""
        parts = num_str.split('/')
        return str(int(re.sub(r'[^0-9]', '', parts[0]))) if parts[0].isdigit() else parts[0]

    def _load_database(self):
        print(f"Loading local card database from {self.cards_dir}...")
        files = glob.glob(os.path.join(self.cards_dir, "en", "*.json"))
        
        count = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    cards = json.load(fp)
                    for card in cards:
                        name = card.get('name', '')
                        number = card.get('number', '')
                        
                        name_clean = self._clean_str(name)
                        number_clean = self._clean_number(number)
                        
                        # Add Source Set ID
                        card['_set_id'] = os.path.basename(f).replace('.json', '')
                        
                        # Index by (Name, Number)
                        key = (name_clean, number_clean)
                        if key not in self.index_name_number:
                            self.index_name_number[key] = []
                        self.index_name_number[key].append(card)
                        
                        # Index by Name
                        if name_clean not in self.index_name_only:
                            self.index_name_only[name_clean] = []
                        self.index_name_only[name_clean].append(card)
                        
                        count += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        print(f"Loaded {count} cards into index.")

    def search_card(self, name, number=None):
        name_clean = self._clean_str(name)
        results = []
        
        # 1. Try Specific (Name + Number)
        if number:
            number_clean = self._clean_number(number)
            key = (name_clean, number_clean)
            if key in self.index_name_number:
                results.extend(self.index_name_number[key])
        
        # 2. If no specific match or no number, try Name only
        if not results and name_clean in self.index_name_only:
             # If we have a number but it didn't match, maybe we can fuzzy match the number?
             # For now, just return all name matches if specific failed? 
             # Or strict? Let's strict for now, but fallback if number was "000/0".
             results.extend(self.index_name_only[name_clean])
             
        return results

if __name__ == "__main__":
    # Test
    db = LocalCardDatabase("cards") # Assuming 'cards' is in CWD
    
    print("\n--- Search: Excadrill 054 ---")
    matches = db.search_card("Excadrill", "054/086")
    for m in matches:
        print(f"Found: {m['name']} #{m['number']} ({m['id']})")

    print("\n--- Search: Charizard (No Num) ---")
    matches = db.search_card("Charizard")
    print(f"Found {len(matches)} Charizards")
