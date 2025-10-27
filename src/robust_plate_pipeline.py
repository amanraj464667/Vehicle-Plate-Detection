import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from enhanced_plate_detection import EnhancedPlateDetector, crop_plate as crop_plate_enhanced
from enhanced_ocr import EnhancedOCR
from plate_detection import detect_plate_contour as legacy_detect

class RobustPlatePipeline:
    """
    Robust pipeline that:
    - Generates multiple plate candidates via enhanced contour-based methods at multiple scales
    - Applies OCR to each candidate with preprocessing and scores results
    - Picks the best text candidate using regex- and confidence-based scoring
    """

    def __init__(self, languages: List[str] = ['en'], time_budget_sec: float = 6.0, max_candidates: int = 6):
        self.detector = EnhancedPlateDetector()
        self.ocr = EnhancedOCR(languages, max_variants=5, time_budget_sec=max(3.0, time_budget_sec - 1.0))
        # Scales to search for candidates to improve robustness under various resolutions
        self.scales = [0.9, 1.0, 1.2]
        self.time_budget_sec = time_budget_sec
        self.max_candidates = max_candidates

    def _collect_candidates(self, image: np.ndarray, y_min_frac: float = 0.0) -> List[Tuple[int, int, int, int]]:
        candidates: List[Tuple[int, int, int, int]] = []
        h, w = image.shape[:2]
        for s in self.scales:
            scaled = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
            dets = self.detector.detect_with_multiple_methods(scaled)
            # Map boxes back to original scale
            inv_s = 1.0 / s
            for (x, y, bw, bh) in dets:
                x0 = int(x * inv_s)
                y0 = int(y * inv_s)
                ww = int(bw * inv_s)
                hh = int(bh * inv_s)
                # Clamp to image bounds
                x0 = max(0, min(x0, w - 1))
                y0 = max(0, min(y0, h - 1))
                ww = max(1, min(ww, w - x0))
                hh = max(1, min(hh, h - y0))
                # Apply region constraint if provided (e.g., prefer lower part)
                cy = y0 + hh / 2.0
                if cy < y_min_frac * h:
                    continue
                candidates.append((x0, y0, ww, hh))
        # Deduplicate using the detector's filter method
        candidates = self.detector.filter_candidates(candidates, image.shape)
        # Keep top-N by quality score heuristic (recompute score)
        scored = [
            (self.detector.calculate_quality_score(x, y, w, h, image.shape), (x, y, w, h))
            for (x, y, w, h) in candidates
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [b for _, b in scored[: self.max_candidates]]

    def _strip_left_blue_band(self, img: np.ndarray) -> np.ndarray:
        """Remove left blue band (IND) if present to avoid confusing OCR."""
        try:
            if img is None or img.size == 0:
                return img
            if len(img.shape) == 2:
                return img
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([90, 60, 50])
            upper = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            max_col = int(0.45 * w)
            if max_col < 5:
                return img
            column_means = mask[:, :max_col].mean(axis=0) / 255.0
            cutoff = 0
            below = 0
            for i, v in enumerate(column_means):
                if v < 0.12:
                    below += 1
                else:
                    below = 0
                if below >= 10:
                    cutoff = i + 1
                    break
            # Ensure leftmost columns really contain blue
            if cutoff > 0 and column_means[:10].mean() > 0.35:
                x0 = min(cutoff, int(0.35 * w))
                return img[:, x0:]
            return img
        except Exception:
            return img

    def _crop_inner_white_region(self, img: np.ndarray) -> np.ndarray:
        """Crop to inner white region (remove thick black frame)"""
        try:
            if img is None or img.size == 0:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            # Normalize and threshold bright region
            gray = cv2.equalizeHist(gray)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, -5)
            # Invert to make white area foreground
            inv = cv2.bitwise_not(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return img
            h, w = gray.shape[:2]
            best = max(cnts, key=cv2.contourArea)
            x, y, ww, hh = cv2.boundingRect(best)
            # Sanity: region should be central and reasonably large
            # Only crop if we retain most of the width/height to avoid cutting off characters
            if ww < 0.8 * w or hh < 0.5 * h:
                return img
            pad = 2
            x = max(0, x + pad)
            y = max(0, y + pad)
            ww = min(w - x, ww - 2 * pad)
            hh = min(h - y, hh - 2 * pad)
            if ww <= 0 or hh <= 0:
                return img
            return img[y:y+hh, x:x+ww]
        except Exception:
            return img

    def _refine_box(self, image: np.ndarray, box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
        """Expand a too-narrow box to a plausible plate aspect and add small vertical padding."""
        x,y,w,h = box
        H, W = image.shape[:2]
        # Add symmetric horizontal expansion to target aspect ~4.0
        target_ratio = 4.0
        if h > 0:
            desired_w = int(target_ratio * h)
            if desired_w > w:
                extra = (desired_w - w) // 2
                x = max(0, x - extra)
                w = min(W - x, w + 2*extra)
        # Add fixed padding
        pad_x = int(0.15 * h)
        pad_y = int(0.10 * h)
        x = max(0, x - pad_x); y = max(0, y - pad_y)
        w = min(W - x, w + 2*pad_x); h = min(H - y, h + 2*pad_y)
        return (x,y,w,h)

    def _deblur_variants(self, plate_img: np.ndarray) -> List[np.ndarray]:
        # Rectify then crop inner white region and strip blue band
        base = plate_img
        try:
            base = self._rectify_plate_roi(base)
        except Exception:
            pass
        base = self._crop_inner_white_region(base)
        base = self._strip_left_blue_band(base)
        # Ensure sufficient resolution for OCR
        try:
            bh, bw = base.shape[:2]
            if bh < 80:
                scale = 80.0 / max(1, bh)
                base = cv2.resize(base, (int(bw * scale), 80), interpolation=cv2.INTER_CUBIC)
        except Exception:
            pass
        variants = [base]
        # Unsharp mask
        try:
            blur = cv2.GaussianBlur(base, (0, 0), 1.0)
            sharp = cv2.addWeighted(base, 1.5, blur, -0.5, 0)
            variants.append(sharp)
        except Exception:
            pass
        # Fast denoise (helps compression artifacts)
        try:
            if len(base.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(base, None, 4, 4, 7, 15)
            else:
                denoised = cv2.fastNlMeansDenoising(base, None, 5, 7, 15)
            variants.append(denoised)
        except Exception:
            pass
        return variants[:3]

    def _ocr_full_image_fallback(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR over the full image and pick best plate-like text"""
        try:
            results = self.ocr.reader.readtext(image, allowlist=self.ocr.allowlist)
        except Exception:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        for (_, text, conf) in results:
            cleaned = self.ocr.clean_plate_text(text)
            if not cleaned:
                continue
            score = self.ocr.calculate_text_confidence(cleaned, max(conf, 0.2), image.shape)
            if score > best_conf:
                best_conf = score
                best_text = cleaned
        return best_text, best_conf

    def process_image(self, image: np.ndarray) -> Tuple[str, Optional[Tuple[int, int, int, int]], Optional[np.ndarray], float, Dict[str, Any]]:
        """
        Returns: text, box, plate_img, confidence, info
        """
        if image is None or image.size == 0:
            return "", None, None, 0.0, {"error": "empty image"}

        import time, os
        start = time.time()

        # Optionally downscale very large images to speed processing
        ih, iw = image.shape[:2]
        max_dim = 1280
        if max(ih, iw) > max_dim:
            scale = max_dim / float(max(ih, iw))
            image = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

        # Collect multiple candidate boxes
        candidates = self._collect_candidates(image)

        # Fallback to legacy if no candidates
        method = "enhanced"
        if not candidates:
            box = legacy_detect(image)
            if box is None:
                # As last resort, OCR the full image
                text_full, conf_full = self._ocr_full_image_fallback(image)
                if text_full:
                    return text_full, None, None, conf_full, {"detection_method": "full_image_ocr"}
                return "", None, None, 0.0, {"error": "no candidates"}
            candidates = [box]
            method = "legacy_contour"

        best_text = ""
        best_conf = 0.0
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_plate_img: Optional[np.ndarray] = None

        # Try OCR on each candidate
        for idx, box in enumerate(candidates):
            # Time budget check
            if time.time() - start > self.time_budget_sec:
                break
            # Expand box to typical plate aspect before cropping
            rbox = self._refine_box(image, box)
            plate_img = crop_plate_enhanced(image, rbox)
            for variant in self._deblur_variants(plate_img):
                text, conf = self.ocr.read_plate_enhanced(variant)
                if conf > best_conf and text:
                    best_conf = conf
                    best_text = text
                    best_box = rbox
                    best_plate_img = variant
                if time.time() - start > self.time_budget_sec:
                    break

        # If still no text, try OCR on the biggest candidate with basic EasyOCR
        if not best_text and candidates:
            x, y, w, h = max(candidates, key=lambda b: b[2] * b[3])
            rbox = self._refine_box(image, (x, y, w, h))
            plate_img = crop_plate_enhanced(image, rbox)
            try:
                # Direct easyocr fallback
                results = self.ocr.reader.readtext(plate_img, allowlist=self.ocr.allowlist)
                for (_, text, conf) in results:
                    cleaned = self.ocr.clean_plate_text(text)
                    if cleaned:
                        score = self.ocr.calculate_text_confidence(cleaned, max(conf, 0.2), plate_img.shape)
                        if score > best_conf:
                            best_text, best_conf = cleaned, score
                            best_box, best_plate_img = (x, y, w, h), plate_img
            except Exception:
                pass

        # If result looks like a brand/short token, try lower-region re-detection
        BRAND_BLOCKLIST = {"KIA", "HONDA", "HYUNDAI", "SUZUKI", "TATA", "MAHINDRA", "FORD", "TOYOTA", "MARUTI", "BMW", "AUDI", "BENZ", "MERCEDES", "VOLVO", "JEEP", "SKODA", "RENAULT", "NISSAN", "KTM"}
        if (best_text and (best_text in BRAND_BLOCKLIST or len(best_text) < 6)):
            candidates_low = self._collect_candidates(image, y_min_frac=0.35)
            if candidates_low:
                for box in candidates_low[:self.max_candidates]:
                    plate_img = crop_plate_enhanced(image, box)
                    for variant in self._deblur_variants(plate_img):
                        text, conf = self.ocr.read_plate_enhanced(variant)
                        if conf > best_conf and text and text not in BRAND_BLOCKLIST:
                            best_conf = conf
                            best_text = text
                            best_box = box
                            best_plate_img = variant

        # Also run OCR on full image and choose better plate-like text
        text_full, conf_full = self._ocr_full_image_fallback(image)
        if text_full and (conf_full > max(best_conf, 0) + 5 or len(text_full) >= 6 and conf_full >= best_conf):
            return text_full, None, None, conf_full, {"detection_method": f"{method}+full_image_ocr_prefer"}

        # As final fallback, OCR full image when nothing found
        if not best_text:
            if text_full:
                return text_full, None, None, conf_full, {"detection_method": f"{method}+full_image_ocr"}

        info: Dict[str, Any] = {
            "detection_method": method,
            "num_candidates": len(candidates),
            "confidence": best_conf,
        }

        return best_text, best_box, best_plate_img, best_conf, info
