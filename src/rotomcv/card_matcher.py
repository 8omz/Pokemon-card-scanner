"""
Pokemon Card Matcher - Production Version

Uses Pokemon TCG API to match OCR results to actual cards.
Strategy: Search by name first, then validate with card number.
"""

import requests
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
import time
from typing import Optional, Dict, List

load_dotenv()

class PokemonCardMatcher:
    """Matches OCR data to Pokemon cards using the TCG API"""
    
    BASE_URL = "https://api.pokemontcg.io/v2/cards"
    
    def __init__(self):
        self.api_key = os.getenv('POKEMON_TCG_API_KEY')
        self.headers = {'X-Api-Key': self.api_key} if self.api_key else {}
        self.cache = {}  # Simple in-memory cache
        
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (0.0 to 1.0)"""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def search_by_name(self, name: str, limit: int = 20) -> List[Dict]:
        """
        Search for cards by name.
        
        Strategy: Strip EX/GX/V suffixes and search by base name.
        The card number will disambiguate if needed.
        """
        if not name or len(name) < 3:
            return []
        
        # Strip common suffixes to get base name
        base_name = name
        suffixes = ['EX', 'GX', 'V', 'VMAX', 'VSTAR', 'VUNION', 'ex', 'gx']
        for suffix in suffixes:
            if name.endswith(f' {suffix}'):
                base_name = name[:-len(suffix)-1].strip()
                break
        
        cache_key = f"name:{base_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {"q": f"name:{base_name}", "pageSize": limit}
            response = requests.get(self.BASE_URL, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                cards = data.get('data', [])
                self.cache[cache_key] = cards
                time.sleep(0.05)  # Rate limiting
                return cards
        except Exception as e:
            print(f"API Error: {e}")
        
        return []
    
    def match_card(self, ocr_name: str, ocr_number: str = None) -> Dict:
        """
        Match a card using OCR data.
        
        Returns:
            {
                'matched': bool,
                'confidence': str,
                'card': dict or None,
                'name_similarity': float,
                'number_match': bool
            }
        """
        result = {
            'matched': False,
            'confidence': 'none',
            'card': None,
            'name_similarity': 0.0,
            'number_match': False,
            'candidates': []
        }
        
        if not ocr_name:
            return result
        
        # Search by name
        candidates = self.search_by_name(ocr_name)
        
        if not candidates:
            return result
        
        result['candidates'] = candidates
        
        # Find best match
        best_card = None
        best_score = 0.0
        
        for card in candidates:
            api_name = card.get('name', '')
            api_number = card.get('number', '')
            
            # Calculate name similarity
            name_sim = self.calculate_similarity(ocr_name, api_name)
            
            # Bonus for number match
            number_bonus = 0.0
            if ocr_number and api_number:
                # Extract just the card number (before the /)
                ocr_num = ocr_number.split('/')[0].strip().lstrip('0')
                api_num = api_number.strip().lstrip('0')
                
                if ocr_num == api_num:
                    number_bonus = 0.3  # Big bonus for exact number match
            
            total_score = name_sim + number_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_card = card
                result['name_similarity'] = name_sim
                result['number_match'] = (number_bonus > 0)
        
        # Determine if we have a match
        if best_card and best_score > 0.5:  # Threshold for match
            result['matched'] = True
            result['card'] = best_card
            
            # Confidence levels
            if best_score > 0.9:
                result['confidence'] = 'high'
            elif best_score > 0.7:
                result['confidence'] = 'medium'
            else:
                result['confidence'] = 'low'
        
        return result
    
    def format_result(self, match_result: Dict) -> str:
        """Format match result for display"""
        if not match_result['matched']:
            return "❌ No match found"
        
        card = match_result['card']
        set_info = card.get('set', {})
        
        emoji = {'high': '✅', 'medium': '⚠️', 'low': '❓'}.get(match_result['confidence'], '❌')
        
        name = card.get('name', 'Unknown')
        number = card.get('number', '?')
        total = set_info.get('printedTotal', '?')
        set_name = set_info.get('name', 'Unknown')
        
        return (f"{emoji} {name} #{number}/{total} ({set_name}) "
                f"[Sim: {match_result['name_similarity']:.0%}, Conf: {match_result['confidence']}]")


# Quick test
if __name__ == "__main__":
    matcher = PokemonCardMatcher()
    
    # Test with your actual OCR data
    test_cases = [
        ("Larvesta", "015/086"),
        ("eanpdee", "054/086"),  # Bad OCR
        ("Venipede", "054/086"),  # Good OCR
        ("Beerrte", "026/086"),  # Bad OCR
        ("Beartic", "026/086"),  # Good OCR
        ("Haunter", "084/202"),
    ]
    
    print("=== Testing Card Matcher ===\n")
    
    for ocr_name, ocr_number in test_cases:
        print(f"OCR: '{ocr_name}' #{ocr_number}")
        result = matcher.match_card(ocr_name, ocr_number)
        print(f"  {matcher.format_result(result)}")
        print()
