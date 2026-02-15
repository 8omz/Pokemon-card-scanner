"""
Analyze all pipeline results and match them to real cards
"""
import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from card_matcher import PokemonCardMatcher

def analyze_results():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv = os.path.join(project_root, "data", "pipeline_output", "pipeline_results.csv")
    output_csv = os.path.join(project_root, "data", "pipeline_output", "matched_cards.csv")
    
    matcher = PokemonCardMatcher()
    
    stats = {
        'total': 0,
        'skipped': 0,
        'matched': 0,
        'high': 0,
        'medium': 0,
        'low': 0,
        'no_match': 0
    }
    
    results = []
    
    print("Analyzing pipeline results...")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total'] += 1
            
            filename = row.get('filename', '')
            status = row.get('status', '')
            ocr_name = row.get('name', '').strip()
            ocr_number = row.get('number', '').strip()
            
            # Skip entries without OCR data
            if status.startswith('skipped') or (not ocr_name and not ocr_number):
                stats['skipped'] += 1
                results.append({
                    'filename': filename,
                    'ocr_name': ocr_name,
                    'ocr_number': ocr_number,
                    'matched': 'SKIPPED',
                    'confidence': 'N/A',
                    'api_name': '',
                    'api_number': '',
                    'api_set': '',
                    'similarity': '0.00'
                })
                continue
            
            # Attempt to match
            match_result = matcher.match_card(ocr_name, ocr_number)
            
            if match_result['matched']:
                stats['matched'] += 1
                card = match_result['card']
                set_info = card.get('set', {})
                
                conf = match_result['confidence']
                stats[conf] += 1
                
                results.append({
                    'filename': filename,
                    'ocr_name': ocr_name,
                    'ocr_number': ocr_number,
                    'matched': 'YES',
                    'confidence': conf.upper(),
                    'api_name': card.get('name', ''),
                    'api_number': f"{card.get('number', '')}/{set_info.get('printedTotal', '')}",
                    'api_set': set_info.get('name', ''),
                    'similarity': f"{match_result['name_similarity']:.2f}"
                })
            else:
                stats['no_match'] += 1
                results.append({
                    'filename': filename,
                    'ocr_name': ocr_name,
                    'ocr_number': ocr_number,
                    'matched': 'NO',
                    'confidence': 'N/A',
                    'api_name': '',
                    'api_number': '',
                    'api_set': '',
                    'similarity': '0.00'
                })
            
            # Progress indicator
            if stats['total'] % 10 == 0:
                print(f"  Processed {stats['total']} entries...")
    
    # Write results
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'ocr_name', 'ocr_number', 'matched', 'confidence',
                      'api_name', 'api_number', 'api_set', 'similarity']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("CARD MATCHING RESULTS")
    print(f"{'='*60}")
    print(f"Total entries: {stats['total']}")
    print(f"Skipped (no OCR data): {stats['skipped']}")
    print(f"Processed: {stats['total'] - stats['skipped']}")
    print(f"\nMatching Results:")
    print(f"  ✅ Matched: {stats['matched']}")
    print(f"     - High confidence: {stats['high']}")
    print(f"     - Medium confidence: {stats['medium']}")
    print(f"     - Low confidence: {stats['low']}")
    print(f"  ❌ No match: {stats['no_match']}")
    
    if stats['total'] - stats['skipped'] > 0:
        match_rate = (stats['matched'] / (stats['total'] - stats['skipped'])) * 100
        print(f"\n📊 Match Rate: {match_rate:.1f}%")
    
    print(f"\n💾 Results saved to: {output_csv}")

if __name__ == "__main__":
    analyze_results()
