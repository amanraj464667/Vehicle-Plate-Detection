import cv2
import numpy as np
import re
from typing import List, Tuple, Optional
import easyocr

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

BLOCKLIST = {"IND", "INDIA", "BHARAT", "IND•", "STATE", "IN", "KIA", "HONDA", "HYUNDAI", "SUZUKI", "TATA", "MAHINDRA", "FORD", "TOYOTA", "MARUTI", "BMW", "AUDI", "BENZ", "MERCEDES", "VOLVO", "JEEP", "SKODA", "RENAULT", "NISSAN", "KTM"}

class EnhancedOCR:
    def __init__(self, languages=['en'], allowlist: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', max_variants: int = 6, time_budget_sec: float = 5.0):
        self.reader = easyocr.Reader(languages, gpu=False)
        self.confidence_threshold = 0.2
        self.allowlist = allowlist
        self.max_variants = max_variants
        self.time_budget_sec = time_budget_sec
        
    def preprocess_plate_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple preprocessing techniques to improve OCR accuracy"""
        preprocessed_images = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Original processed image
        preprocessed_images.append(gray)
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        preprocessed_images.append(clahe_img)
        
        # 2. Gaussian blur + sharpen
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        preprocessed_images.append(sharpened)
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(morph_img)
        
        # 4. Threshold with different methods
        # Otsu threshold
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu_thresh)
        
        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(adaptive_thresh)
        
        # 5. Inverted threshold (white text on black background)
        _, inv_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(inv_thresh)
        
        # 6. Resize images for better OCR (if too small)
        resized_images = []
        for img in preprocessed_images:
            height, width = img.shape[:2]
            if height < 40 or width < 100:
                # Scale up small images
                scale_factor = max(2, 100 // max(1, width), 40 // max(1, height))
                resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Limit number of variants to speed up processing
        return resized_images[: self.max_variants]
    
    def clean_plate_text(self, text: str) -> str:
        """Clean and standardize plate text"""
        if not text:
            return ""
        
        # Remove spaces and convert to uppercase
        cleaned = text.replace(" ", "").replace("-", "").upper()
        
        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        
        # Common OCR corrections for license plates
        corrections = {
            'O': '0',  # O to 0
            'I': '1',  # I to 1
            'S': '5',  # S to 5 (sometimes)
            'G': '6',  # G to 6 (sometimes)
            'B': '8',  # B to 8 (sometimes)
        }
        
        # Apply corrections intelligently (only if it makes sense in context)
        # For Indian plates: typically 2-4 letters + 4 digits or similar patterns
        if len(cleaned) >= 4:
            # If last 4 characters should be numbers, apply number corrections
            last_part = cleaned[-4:]
            corrected_last_part = ""
            for char in last_part:
                if char in corrections and char in ['O', 'I']:
                    corrected_last_part += corrections[char]
                else:
                    corrected_last_part += char
            
            # If first part should be letters, avoid number corrections there
            first_part = cleaned[:-4]
            cleaned = first_part + corrected_last_part
        
        return cleaned
    
    def calculate_text_confidence(self, text: str, confidence: float, image_shape: Tuple) -> float:
        """Calculate overall confidence score for detected text"""
        if not text or confidence < 0.1:
            return 0.0
        
        # Discard blocklisted side-text
        if text in BLOCKLIST:
            return 0.0
        
        score = confidence * 100
        
        # Length bonus (Indian plates are typically 8-10 characters)
        if 7 <= len(text) <= 10:
            score += 30
        elif 6 <= len(text) <= 12:
            score += 20
        elif 4 <= len(text) <= 15:
            score += 10
        else:
            score -= 20
        
        # Mix of letters and digits bonus
        has_alpha = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        if has_alpha and has_digit:
            score += 15
        
        # Pattern matching bonus for Indian plate patterns
        indian_patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$',  # Standard format
            r'^[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}$',         # Alternative format
            r'^[A-Z]{2}[0-9]{6}$',                       # Number heavy format
        ]
        
        for pattern in indian_patterns:
            if re.match(pattern, text):
                score += 30
                break
        else:
            # Partial pattern matching
            if re.search(r'[A-Z]{2,4}[0-9]{3,4}', text):
                score += 15
            elif re.search(r'[A-Z]+[0-9]+', text):
                score += 10
        
        # Penalize very short or very long text
        if len(text) < 5:
            score -= 30
        elif len(text) > 12:
            score -= 15
        
        return max(0, min(100, score))
    
    def _assemble_from_detections(self, results, img_shape: Tuple) -> Tuple[str, float]:
        """Assemble full plate from multiple EasyOCR word detections"""
        if not results:
            return "", 0.0
        # Filter out left-side region (blue band with IND) and blocklist
        width = img_shape[1]
        filtered = []
        centers_y = []
        for (bbox, text, conf) in results:
            cleaned = self.clean_plate_text(text)
            if not cleaned or cleaned in BLOCKLIST:
                continue
            # bbox is list of 4 points [[x1,y1]..]
            try:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = sum(xs) / 4.0
                cy = sum(ys) / 4.0
                box_h = max(ys) - min(ys)
            except Exception:
                cx = width / 2.0
                cy = 0.5 * img_shape[0]
                box_h = 0
            centers_y.append(cy)
            filtered.append((bbox, cleaned, float(conf), box_h, cx, cy))
        if not filtered:
            return "", 0.0
        # Keep tokens aligned to the main text line
        if centers_y:
            import statistics
            median_y = statistics.median(centers_y)
            aligned = []
            for (bbox, t, c, h, cx, cy) in filtered:
                if abs(cy - median_y) <= 0.28 * img_shape[0]:
                    aligned.append((bbox, t, c, h, cx))
            filtered = aligned if aligned else [(b, t, c, h, cx) for (b, t, c, h, cx, cy) in filtered]
        # Drop tokens in leftmost 25% unless very confident and long
        filtered = [r for r in filtered if not (r[4] < 0.25 * width and len(r[1]) <= 4 and r[2] < 0.8)]
        if not filtered:
            return "", 0.0
        # Sort by x and join text (prefer taller tokens)
        filtered.sort(key=lambda r: (min(p[0] for p in r[0]), -r[3]))
        parts = [t for _, t, _, _, _ in filtered]
        joined = ''.join(parts)
        # Weighted confidence by token length
        weights = [max(1, len(t)) for _, t, _, _, _ in filtered]
        confs = [c for _, _, c, _, _ in filtered]
        base_conf = sum(w * c for w, c in zip(weights, confs)) / float(sum(weights))
        # Pick best substring that matches plate-like patterns
        candidates = [joined]
        L = len(joined)
        for a in range(0, max(1, L - 5)):
            for b in range(a + 6, min(L, a + 12) + 1):
                candidates.append(joined[a:b])
        best_text = ""
        best_score = 0.0
        for cand in candidates:
            score = self.calculate_text_confidence(cand, max(base_conf, 0.2), img_shape)
            if score > best_score:
                best_score = score
                best_text = cand
        return best_text, best_score
    
    def _tesseract_read(self, img: np.ndarray) -> Tuple[str, float]:
        if not _HAS_TESSERACT or img is None or img.size == 0:
            return "", 0.0
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            # binarize
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            raw = pytesseract.image_to_string(th, config=config)
            cleaned = self.clean_plate_text(raw)
            if not cleaned:
                return "", 0.0
            # Estimate confidence using heuristic since pytesseract confidence is not returned here
            conf = 0.6
            score = self.calculate_text_confidence(cleaned, conf, img.shape)
            return cleaned, score
        except Exception:
            return "", 0.0

    def read_plate_enhanced(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Enhanced plate reading with multiple preprocessing attempts"""
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        
        import time
        start = time.time()
        
        # Get multiple preprocessed versions
        preprocessed_images = self.preprocess_plate_image(plate_image)
        
        best_text = ""
        best_confidence = 0.0
        
        for img in preprocessed_images:
            # Time budget check
            if time.time() - start > self.time_budget_sec:
                break
            try:
                # Run OCR with allowlist to speed up and reduce noise
                results = self.reader.readtext(img, allowlist=self.allowlist)
                # Assemble full plate string from multiple boxes
                joined, enhanced_conf = self._assemble_from_detections(results, img.shape)
                if joined and enhanced_conf > best_confidence:
                    best_text, best_confidence = joined, enhanced_conf
                    if best_confidence >= 90:
                        return best_text, min(100, best_confidence)
            except Exception as e:
                print(f"OCR processing error: {e}")
                continue
        
        # Extra fallbacks if still empty
        if not best_text:
            try:
                results = self.reader.readtext(plate_image, allowlist=self.allowlist)
                joined, enhanced_conf = self._assemble_from_detections(results, plate_image.shape)
                if joined and enhanced_conf > best_confidence:
                    best_text, best_confidence = joined, enhanced_conf
            except Exception:
                pass
            try:
                inv = cv2.bitwise_not(plate_image if len(plate_image.shape)==2 else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY))
                results = self.reader.readtext(inv, allowlist=self.allowlist)
                joined, enhanced_conf = self._assemble_from_detections(results, inv.shape)
                if joined and enhanced_conf > best_confidence:
                    best_text, best_confidence = joined, enhanced_conf
            except Exception:
                pass
            # Tesseract fallback
            if _HAS_TESSERACT:
                cand, score = self._tesseract_read(plate_image)
                if score > best_confidence:
                    best_text, best_confidence = cand, score
        
        # Final validation
        if best_text and len(best_text) >= 5:
            return best_text, min(100, best_confidence)
        else:
            return best_text if best_text else "", min(100, best_confidence) if best_text else 0.0
    
    def texts_similar(self, text1: str, text2: str, max_diff: int = 2) -> bool:
        """Check if two texts are similar (allowing for OCR errors)"""
        if not text1 or not text2:
            return False
        
        if abs(len(text1) - len(text2)) > max_diff:
            return False
        
        # Calculate edit distance
        def edit_distance(s1, s2):
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            
            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                distances_ = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        distances_.append(distances[i1])
                    else:
                        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                distances = distances_
            return distances[-1]
        
        return edit_distance(text1, text2) <= max_diff

# Enhanced OCR function for compatibility
def read_plate_with_confidence(plate_image: np.ndarray) -> Tuple[str, float]:
    """Read plate with confidence score"""
    ocr = EnhancedOCR()
    return ocr.read_plate_enhanced(plate_image)