import re
import os
from src.rotomcv.db_client import RotomDB

class LocalCardDatabase:
    def __init__(self, cards_dir=None):
        # cards_dir is legacy argument, ignored now
        self.db = RotomDB()
        print("LocalCardDatabase: Connected to MongoDB via RotomDB.")
        
        # Check if DB is empty, warn user if so
        cnt = self.db.count_cards()
        if cnt == 0:
            print("WARNING: MongoDB 'cards' collection is empty. Please run 'src/rotomcv/ingest_cards.py'.")
        else:
            print(f"Database contains {cnt} cards.")

    def _clean_str(self, s):
        if not s: return ""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _clean_number(self, num_str):
        # Handle "015/086" -> "15"
        if not num_str: return ""
        parts = num_str.split('/')
        val = parts[0]
        if val.isdigit():
            return str(int(val))
        return val

    def search_card(self, name, number=None):
        """
        Search using MongoDB.
        """
        name_clean = self._clean_str(name)
        results = []
        
        # 1. Exact Name + Number
        if number:
            number_clean = self._clean_number(number)
            # MongoDB Query: Number is "15", Name is "Celebi"
            # We want cards with number_clean == "15"
            potential_cards = list(self.db.cards.find({"number_clean": number_clean}).limit(50))
            
            import difflib
            
            for card in potential_cards:
                # Check Name similarity
                card_name_clean = card.get("name_clean", "")
                
                # Exact match
                if name_clean == card_name_clean:
                    results.append(card)
                    continue
                
                # Fuzzy match
                # Check if name is subset or high ratio
                if name_clean in card_name_clean or card_name_clean in name_clean:
                    results.append(card)
                    continue
                
                ratio = difflib.SequenceMatcher(None, name_clean, card_name_clean).ratio()
                if ratio > 0.6:
                    results.append(card)
            
            if results:
                return results

        # 2. Name Only Search (if no number or no match found)
        # Use regex for starts_with or simple text search
        # Escape regex special chars
        safe_name = re.escape(name_clean)
        # Search for names starting with query
        cursor = self.db.cards.find({"name_clean": {"$regex": f"^{safe_name}", "$options": "i"}}).limit(20)
        results.extend(list(cursor))
        
        if not results:
             # Try looser search (contains)
             cursor = self.db.cards.find({"name_clean": {"$regex": safe_name, "$options": "i"}}).limit(20)
             results.extend(list(cursor))
             
        return results

if __name__ == "__main__":
    # Test
    # Assuming standard usage without cards_dir arg needed anymore
    db = LocalCardDatabase() 
    
    print("\n--- Search: Excadrill 054 ---")
    matches = db.search_card("Excadrill", "054/086")
    for m in matches:
        print(f"Found: {m.get('name')} #{m.get('number')} ({m.get('id')})")

    print("\n--- Search: Charizard (No Num) ---")
    matches = db.search_card("Charizard")
    print(f"Found {len(matches)} Charizards")
