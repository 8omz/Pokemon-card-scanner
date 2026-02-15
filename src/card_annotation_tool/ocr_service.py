import re
import cv2
import numpy as np
import os
import sys
import csv
import pytesseract

# Regex Patterns for Card Numbers
# Regex Patterns for Card Numbers
NUMBER_PATTERNS = [
    r'\d{1,3}\s*/\s*\d{1,3}[a-zA-Z]?',       # Standard: 015/102
    r'(?:TG|SV|SWSH|RC|XY|BW|SM|GG|BSP)\s*\d{1,3}', # Explicit Prefixes
    r'(?:TG|SV|SWSH|RC|XY|BW|SM|GG|BSP)\s*\d{1,3}\s*/\s*\d{1,3}', # Prefix with denominator
    r'^\d{2,3}$'                # Just digits (Strict fallback: 2-3 digits to avoid '1')
]

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
        Initialize the PaddleOCR engine with accuracy-optimized settings.
        """
        print("Initializing PokemonCardOCR engine...")
        # PP-OCRv4 mobile — best balance for small pre-cropped ROI images
        # v5 tested but produced more noise (HP values, energy symbols as text)
        # use_textline_orientation=False: Rectifier guarantees upright images
        self.ocr = PaddleOCR(
            use_textline_orientation=False,
            lang='en',
            enable_mkldnn=False,
            ocr_version='PP-OCRv4',
        )

        # Load Pokemon name dictionary for validation & fuzzy matching
        self.pokemon_names = {}  # lowercase -> original case
        names_csv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "manifests", "pokemon_names.csv"
        )
        if os.path.exists(names_csv):
            with open(names_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['Name'].strip()
                    self.pokemon_names[name.lower()] = name
            print(f"Loaded {len(self.pokemon_names)} Pokemon names for matching.")
        else:
            print(f"WARNING: Pokemon names CSV not found at {names_csv}")

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

    def preprocess_default(self, image):
        """
        Strategy 1: CLAHE + Bilateral Filter (original — best for clean/matte cards).
        """
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        contrast = cv2.bilateralFilter(contrast, 9, 75, 75)
        contrast = cv2.copyMakeBorder(contrast, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    def preprocess_sharpen(self, image):
        """
        Strategy 2: Sharpen + Otsu threshold (best for holo/reflective surfaces).
        """
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Sharpen to cut through holo reflections
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(gray, -1, kernel)
        # Otsu binarization — auto-finds optimal threshold
        _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.copyMakeBorder(binary, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255])
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def preprocess_adaptive(self, image):
        """
        Strategy 3: Adaptive threshold (best for uneven lighting/zoomed-in shots).
        """
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Adaptive threshold handles uneven lighting across the ROI
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        binary = cv2.copyMakeBorder(binary, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255])
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def preprocess_variants(self, image):
        """
        Yields (name, image) tuples for different preprocessing strategies.
        Used for robust number extraction.
        """
        if image is None:
            return

        # 1. Original (Grayscale + Upscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Upscale for better Tesseract recognition on small text
        scale = 3
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield "normal", upscaled

        # 2. Inverted (White text on dark bg -> Black on white)
        inverted = cv2.bitwise_not(upscaled)
        yield "inverted", inverted

        # 3. Thresholded (High contrast)
        # Simple binary threshold
        _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        yield "threshold", thresh
        
        # 4. Inverted Threshold
        yield "threshold_inv", cv2.bitwise_not(thresh)

    def clean_pokemon_name(self, text):
        """
        Clean and normalize Pokemon names, handling special suffixes.
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # 1. Strip all digits - numbers are never part of a Pokemon name
        text = re.sub(r'\d+', '', text)
        
        # 2. Clean up whitespace
        text = ' '.join(text.split())
        
        # 3. Strip stray single-character words (OCR fragments)
        # Keep "V" since it's a valid Pokemon suffix
        words = text.split()
        words = [w for w in words if len(w) > 1 or w.upper() == 'V']
        text = ' '.join(words)
        
        if not text:
            return ""
        
        # Known Pokemon card suffixes (order matters - check longer ones first)
        suffixes = [
            'VSTAR', 'VMAX', 'VUNION', 
            'GX', 'EX', 'V',
            'ex', 'gx',  # Lowercase variants
            '-EX', '-GX', '-V'  # Hyphenated variants
        ]
        
        # Check if suffix is already properly separated
        for suffix in suffixes:
            if text.endswith(f' {suffix}'):
                return text  # Already clean
            if text.endswith(f'-{suffix}'):
                return text.replace(f'-{suffix}', f' {suffix}')
        
        # Check for concatenated suffixes (case-insensitive)
        text_lower = text.lower()
        for suffix in suffixes:
            suffix_lower = suffix.lower()
            if text_lower.endswith(suffix_lower):
                base_name = text[:-len(suffix)]
                return f"{base_name} {suffix.upper()}"
        
        return text
    
    def clean_card_number(self, text):
        """
        Extract ONLY the card number pattern (e.g., "054/086") from OCR text.
        """
        if not text:
            return ""
        
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('l', '1').replace('I', '1')
        
        match = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', text)
        if match:
            card_num = match.group(1).zfill(3)
            set_total = match.group(2)
            return f"{card_num}/{set_total}"
        
        digits = re.findall(r'\d+', text)
        if len(digits) >= 2:
            card_num = digits[0].zfill(3)
            set_total = digits[1]
            return f"{card_num}/{set_total}"
        
        return ""

    def extract_text_for_number(self, crops_dict, engine='tesseract'):
        """
        Robustly extracts card number by trying multiple ROIs and preprocessing methods.
        Iterates through:
          ROIs: card_number_bl, card_number_br, card_number_center
          Variants: normal, inverted, threshold, threshold_inv
          Configs: PSM 6, 7, 8
        """
        # Prioritize ROIs
        rois_to_check = ['card_number_bl', 'card_number_br', 'card_number_center']
        # Fallback to generic 'card_number' if present (backwards compat)
        if 'card_number' in crops_dict and 'card_number_bl' not in crops_dict:
             rois_to_check.insert(0, 'card_number')

        best_guess = ""
        
        for roi_key in rois_to_check:
            crop = crops_dict.get(roi_key)
            if crop is None:
                continue
                
            # Iterate image variants
            for variant_name, processed_img in self.preprocess_variants(crop):
                
                # --- PASS 1: Strict Digits Only (Best for standard sets) ---
                # Whitelist: Digits, /, -
                whitelist_digits = "0123456789/-"
                config_digits = f'--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist_digits}'
                
                text = pytesseract.image_to_string(processed_img, config=config_digits).strip()
                text = text.replace(' ', '')
                # No substitutions needed for digits pass (O/I/L shouldn't exist)
                
                # Check for standard slash pattern 000/000
                match = re.search(r'\d{1,3}/\d{1,3}', text)
                if match:
                    found_text = match.group(0)
                    print(f"  [SUCCESS] Found number '{found_text}' in {roi_key} ({variant_name}) [DIGITS PASS]")
                    return found_text

                # --- PASS 2: Alphanumeric (For TG, SV, etc) ---
                # Whitelist: Digits, /, -, and specific set prefix letters
                whitelist_alpha = "0123456789/TGSHWRCVPXYBM-"
                config_alpha = f'--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist_alpha}'
                
                text = pytesseract.image_to_string(processed_img, config=config_alpha).strip()
                
                # Clean common OCR errors
                text_upper = text.upper().replace(' ', '')
                text_upper = text_upper.replace('O', '0').replace('I', '1').replace('L', '1')
                
                if text_upper and len(text_upper) < 20: 
                        # print(f"    [DEBUG] {roi_key} ({variant_name}): '{text_upper}'")
                        pass
                
                for pattern in NUMBER_PATTERNS:
                        match = re.search(pattern, text_upper)
                        if match:
                            found_text = match.group(0)
                            print(f"  [SUCCESS] Found number '{found_text}' in {roi_key} ({variant_name}) [ALPHA PASS]")
                            return found_text
                
                # Store best guess if it looks kinda like a number (contains /)
                if '/' in text_upper and len(text_upper) < 10:
                    best_guess = text_upper

        return best_guess

    def _ocr_raw(self, processed_img):
        """
        Run OCR and return (text, confidence) tuple.
        Returns ("", 0.0) on failure.
        """
        try:
            result = self.ocr.predict(processed_img)
            if result and isinstance(result, list) and len(result) > 0:
                res = result[0]
                
                # Handle Paddlex Dict Format
                if isinstance(res, dict) and 'rec_texts' in res:
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    full_text = " ".join(texts)
                    avg_conf = sum(scores) / len(scores) if scores else 0.0
                    return full_text, avg_conf
                
                # Handle Standard List Format
                if isinstance(res, list):
                    texts = []
                    confs = []
                    for line in res:
                        if len(line) >= 2:
                            texts.append(line[0])
                            confs.append(line[1])
                    full_text = " ".join(texts)
                    avg_conf = sum(confs) / len(confs) if confs else 0.0
                    return full_text, avg_conf
                    
        except Exception as e:
            print(f"OCR Error: {e}")
        return "", 0.0

    def _match_pokemon_name(self, cleaned_name):
        """
        Check if cleaned name matches a known Pokemon.
        Returns (matched_name, is_exact) or (None, False).
        """
        if not cleaned_name:
            return None, False
        
        # Strip suffix for matching (e.g. "Serperior EX" -> check "Serperior")
        base_name = cleaned_name.split()[0] if cleaned_name.split() else cleaned_name
        
        # Exact match (case-insensitive)
        if base_name.lower() in self.pokemon_names:
            return self.pokemon_names[base_name.lower()], True
        
        return None, False

    def _get_fuzzy_match(self, cleaned_name):
        """
        Get the best fuzzy match and its similarity ratio.
        Returns (matched_name, ratio).
        """
        if not cleaned_name or not self.pokemon_names:
            return None, 0.0
        
        parts = cleaned_name.split()
        base_name = parts[0]
        suffix = ' '.join(parts[1:]) if len(parts) > 1 else ''
        
        matches = get_close_matches(base_name.lower(), self.pokemon_names.keys(), n=1, cutoff=0.5)
        if matches:
            match_key = matches[0]
            # Calculate actual ratio for decision making
            ratio = 0.0
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, base_name.lower(), match_key).ratio()
            
            corrected = self.pokemon_names[match_key]
            if suffix:
                full_name = f"{corrected} {suffix}"
                return full_name, ratio
            return corrected, ratio
            
        return None, 0.0

    def _ocr_tesseract(self, processed_img):
        """
        Run Tesseract OCR with settings tuned for Pokemon cards.
        - Precision scaling (2x)
        - PSM 8 (Single Word) - CRITICAL for single-line names on cards
        """
        try:
            # Tesseract performs better on larger text
            # Upscale by 2x
            h, w = processed_img.shape[:2]
            scaled = cv2.resize(processed_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            
            # Tesseract expects RGB (cv2 is BGR)
            rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            
            # config='--psm 8' treats the image as a single word.
            # This worked best in tuning for "Excadrill".
            text = pytesseract.image_to_string(rgb, config='--psm 8').strip()
            
            if text:
                return text, 0.8 # Placeholder confidence
        except Exception as e:
            print(f"Tesseract Error: {e}")
            
        return "", 0.0

    def extract_text(self, roi_crop, is_number=False, engine='hybrid'):
        """
        Main API method — tries up to 3 preprocessing strategies.
        'hybrid' engine: Tries PaddleOCR first, falls back to Tesseract if fails.
        """
        if roi_crop is None:
            return ""

        # --- Helper to run a specific engine on all strategies ---
        def _run_strategies_with_engine(target_engine):
            strategies = [
                ("default", self.preprocess_default),
                ("sharpen", self.preprocess_sharpen),
                ("adaptive", self.preprocess_adaptive),
            ]
            best_t = ""
            best_c = 0.0
            
            for name, preprocess_fn in strategies:
                processed_img = preprocess_fn(roi_crop)
                if processed_img is None:
                    continue
                
                if target_engine == 'tesseract':
                    raw_text, confidence = self._ocr_tesseract(processed_img)
                else:
                    raw_text, confidence = self._ocr_raw(processed_img)
                
                # If number, we just want digits
                if is_number:
                     if confidence > best_c:
                        best_c = confidence
                        best_t = raw_text
                else: 
                    # If name, clean and check match
                    cleaned = self.clean_pokemon_name(raw_text)
                    if confidence > best_c:
                        best_c = confidence
                        best_t = cleaned
                    
                    # Stop if we find a good match (only for Paddle, Tesseract is one-shot fallback usually)
                    # But for consistency, we check here too.
                    match, is_exact = self._match_pokemon_name(cleaned)
                    if is_exact:
                        return cleaned, 1.0, True # Found exact
                    
                    fuzzy, ratio = self._get_fuzzy_match(cleaned)
                    if fuzzy and ratio >= 0.8:
                         return fuzzy, ratio, True # Found strong fuzzy
            
            return best_t, best_c, False

        # --- Execution Logic ---
        
        # 1. Card Numbers (Simple, use Paddle usually)
        if is_number:
             # Just use paddle for numbers unless forced otherwise
             # Tesseract PSM 8 might be bad for "123/456" (symbols).
             # Let's stick to Paddle for numbers for now as it was working fine.
             text, conf, found = _run_strategies_with_engine('paddle')
             return self.clean_card_number(text)

        # 2. Names (Hybrid Logic)
        
        # Step A: Try Paddle (Primary)
        if engine == 'hybrid' or engine == 'paddle':
            paddle_text, paddle_conf, paddle_found = _run_strategies_with_engine('paddle')
            if paddle_found:
                # Re-verify match string formatting
                match, is_exact = self._match_pokemon_name(paddle_text)
                if is_exact: return match
                fuzzy, ratio = self._get_fuzzy_match(paddle_text)
                if fuzzy: return fuzzy
                return paddle_text # Should be unreachable if paddle_found is true
            
            # If Paddle failed to find a Match, check criteria for Fallback
            # If confidence is consistently low OR text is empty -> needs fallback
            # If engine is ONLY paddle, we stop here.
            if engine == 'paddle':
                # Return best effort
                final_fuzzy, final_ratio = self._get_fuzzy_match(paddle_text)
                if final_fuzzy and final_ratio >= 0.8: return final_fuzzy
                return paddle_text

        # Step B: Try Tesseract (Fallback or Primary)
        if engine == 'hybrid' or engine == 'tesseract':
            # Only run if we are in hybrid mode (and Paddle failed) OR properly Tesseract mode
            
            # For hybrid, we only run if Paddle didn't return a strong match.
            # (Which is true if we reached here).
            
            tess_text, tess_conf, tess_found = _run_strategies_with_engine('tesseract')
            
            if tess_found:
                match, is_exact = self._match_pokemon_name(tess_text)
                if is_exact: return match
                fuzzy, ratio = self._get_fuzzy_match(tess_text)
                if fuzzy: return fuzzy

            # Step C: Compare Best Efforts (if both failed to find "Match")
            # If we are here, neither engine found a dictionary match.
            # We must return the "best garbage" or empty.
            
            # Prioritize Paddle's garbage usually, unless it's empty.
            if engine == 'hybrid':
                # If Paddle captured something decent (conf > 0.5?), might be valid unknown name
                # If Tesseract captured something, might be better?
                # Hard to compare confidence between engines.
                # Heuristic: If Paddle is Empty, use Tesseract.
                if not paddle_text.strip() and tess_text.strip():
                    return tess_text
                
                # Check for "Excadrill" specific case? No, general logic.
                # If Tesseract result has a fuzzy match < 0.8 but > Paddle?
                pass

        # Final Fallback
        # If hybrid, we return Paddle's result by default if Tesseract didn't find a KNOWN match.
        # Why? Because Tesseract produced 50% less text in general. 
        # So Paddle's "garbage" is more likely to be real text than Tesseract's "garbage".
        if engine == 'hybrid':
            # Check Tesseract one last time for fuzzy
            f, r = self._get_fuzzy_match(tess_text)
            if f and r > 0.6: # Lower threshold for fallback consideration?
                 return f
            return paddle_text
            
        if engine == 'tesseract':
            return tess_text
            
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
