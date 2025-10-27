import cv2
import numpy as np
import imutils
from typing import List, Tuple, Optional

class EnhancedPlateDetector:
    def __init__(self):
        self.min_plate_area = 500
        self.max_plate_area = 50000
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced image preprocessing for better plate detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return gray, blurred
    
    def detect_with_multiple_methods(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Use multiple detection methods and combine results"""
        candidates = []
        
        # Method 1: Enhanced contour detection
        candidates.extend(self.detect_contour_method(image))
        
        # Method 2: Morphological operations
        candidates.extend(self.detect_morphological_method(image))
        
        # Method 3: Edge-based detection
        candidates.extend(self.detect_edge_based_method(image))
        
        # Filter and rank candidates
        filtered_candidates = self.filter_candidates(candidates, image.shape)
        
        return filtered_candidates
    
    def detect_contour_method(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Enhanced contour-based detection"""
        gray, blurred = self.preprocess_image(image)
        
        candidates = []
        
        # Try multiple edge detection parameters
        edge_params = [(50, 150), (100, 200), (30, 100), (80, 160)]
        
        for low_thresh, high_thresh in edge_params:
            # Edge detection
            edges = cv2.Canny(blurred, low_thresh, high_thresh)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                if self.is_valid_plate_candidate(x, y, w, h, image.shape):
                    candidates.append((x, y, w, h))
        
        return candidates
    
    def detect_morphological_method(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Morphological operations based detection"""
        gray, blurred = self.preprocess_image(image)
        
        candidates = []
        
        # Apply different morphological operations
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        ]
        
        for kernel in kernels:
            # BlackHat operation to highlight dark regions on light background
            blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if self.is_valid_plate_candidate(x, y, w, h, image.shape):
                    candidates.append((x, y, w, h))
        
        return candidates
    
    def detect_edge_based_method(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Edge-based detection with Hough lines"""
        gray, blurred = self.preprocess_image(image)
        
        candidates = []
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Group lines to find rectangular regions
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if -15 <= angle <= 15 or 165 <= angle <= 195:  # Horizontal lines
                    horizontal_lines.append(line[0])
                elif 75 <= angle <= 105 or -105 <= angle <= -75:  # Vertical lines
                    vertical_lines.append(line[0])
            
            # Find intersections and create candidate rectangles
            for h_line in horizontal_lines[:10]:
                for v_line in vertical_lines[:10]:
                    # Simplified intersection logic
                    x1h, y1h, x2h, y2h = h_line
                    x1v, y1v, x2v, y2v = v_line
                    
                    # Create potential rectangle
                    x = min(x1h, x2h, x1v, x2v)
                    y = min(y1h, y2h, y1v, y2v)
                    w = max(x1h, x2h, x1v, x2v) - x
                    h = max(y1h, y2h, y1v, y2v) - y
                    
                    if self.is_valid_plate_candidate(x, y, w, h, image.shape):
                        candidates.append((x, y, w, h))
        
        return candidates
    
    def is_valid_plate_candidate(self, x: int, y: int, w: int, h: int, image_shape: Tuple) -> bool:
        """Validate if a rectangle is a potential license plate"""
        img_height, img_width = image_shape[:2]
        
        # Area checks
        area = w * h
        if area < self.min_plate_area or area > self.max_plate_area:
            return False
        
        # Aspect ratio checks (license plates are typically wider than tall)
        aspect_ratio = w / float(h)
        if aspect_ratio < 1.5 or aspect_ratio > 6.5:
            return False
        
        # Size relative to image
        if w < 0.05 * img_width or w > 0.6 * img_width:
            return False
        if h < 0.02 * img_height or h > 0.3 * img_height:
            return False
        
        # Position checks (plates are usually not at the very edges)
        # Relaxed margin and correct right/bottom edge checks to use x+w and y+h
        margin = 0.02
        if (x < margin * img_width or (x + w) > (1 - margin) * img_width or
            y < margin * img_height or (y + h) > (1 - margin) * img_height):
            # Only reject if the box significantly crosses image border
            # Allow slight touching of edges
            edge_overlap = (x < 0 or y < 0 or (x + w) > img_width or (y + h) > img_height)
            if edge_overlap:
                return False
        
        return True
    
    def filter_candidates(self, candidates: List[Tuple[int, int, int, int]], 
                         image_shape: Tuple) -> List[Tuple[int, int, int, int]]:
        """Filter and rank candidates by quality"""
        if not candidates:
            return []
        
        # Remove duplicates (merge overlapping rectangles)
        filtered = []
        for x, y, w, h in candidates:
            is_duplicate = False
            for fx, fy, fw, fh in filtered:
                # Calculate overlap
                overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
                overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
                overlap_area = overlap_x * overlap_y
                
                area1 = w * h
                area2 = fw * fh
                
                # If overlap is significant, it's a duplicate
                if overlap_area > 0.5 * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append((x, y, w, h))
        
        # Rank by quality score
        scored_candidates = []
        for x, y, w, h in filtered:
            score = self.calculate_quality_score(x, y, w, h, image_shape)
            scored_candidates.append(((x, y, w, h), score))
        
        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return [candidate[0] for candidate in scored_candidates[:5]]
    
    def calculate_quality_score(self, x: int, y: int, w: int, h: int, 
                               image_shape: Tuple) -> float:
        """Calculate quality score for a plate candidate"""
        img_height, img_width = image_shape[:2]
        
        score = 0.0
        
        # Aspect ratio score (prefer ratios around 3:1 to 4:1)
        aspect_ratio = w / float(h)
        ideal_ratio = 3.5
        ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
        score += ratio_score * 30
        
        # Size score (prefer medium-sized plates)
        area = w * h
        relative_area = area / (img_width * img_height)
        if 0.01 <= relative_area <= 0.05:
            score += 25
        elif 0.005 <= relative_area <= 0.1:
            score += 15
        
        # Position score (prefer plates in lower 2/3 of image)
        center_y = y + h // 2
        if center_y > img_height * 0.3:
            score += 20
        
        # Width score (prefer plates that are not too thin or too wide)
        relative_width = w / img_width
        if 0.1 <= relative_width <= 0.4:
            score += 15
        
        return score
    
    def detect_best_plate(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Main detection method that returns the best plate candidate"""
        candidates = self.detect_with_multiple_methods(image)
        
        if candidates:
            return candidates[0]  # Return the best candidate
        
        return None

def detect_plate_enhanced(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Enhanced plate detection function"""
    detector = EnhancedPlateDetector()
    return detector.detect_best_plate(image)

def crop_plate(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop plate from image with padding"""
    x, y, w, h = box
    
    # Add small padding
    padding = 12
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    cropped = image[y:y+h, x:x+w].copy()
    
    # Enhance the cropped plate
    if cropped.size > 0:
        # Convert to grayscale if needed
        if len(cropped.shape) == 3:
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray_cropped = cropped
            
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_cropped = clahe.apply(gray_cropped)
        
        # Convert back to BGR if original was BGR
        if len(cropped.shape) == 3:
            cropped = cv2.cvtColor(enhanced_cropped, cv2.COLOR_GRAY2BGR)
        else:
            cropped = enhanced_cropped
    
    return cropped

# Legacy compatibility functions
def detect_plate_contour(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Legacy function - now uses enhanced detection"""
    return detect_plate_enhanced(image)