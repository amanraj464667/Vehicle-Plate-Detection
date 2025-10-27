
import easyocr
import cv2, os, numpy as np
from pathlib import Path

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

try:
    from src.char_cnn import load_model as load_cnn_model
    _HAS_CNN = True
except Exception:
    _HAS_CNN = False

class OCRReader:
    def __init__(self, backend='easyocr', langs=['en'], cnn_model_path=None):
        self.backend = backend.lower()
        self.cnn_model = None
        if self.backend == 'easyocr':
            self.reader = easyocr.Reader(langs, gpu=False)
        elif self.backend == 'tesseract':
            if not _HAS_TESSERACT:
                raise ImportError('pytesseract is not installed. Please install it with: pip install pytesseract')
            self.reader = None
        elif self.backend == 'cnn':
            if cnn_model_path is None:
                raise ValueError('cnn_model_path must be provided for cnn backend')
            if not _HAS_CNN:
                raise ImportError('char_cnn module not available in src/. Ensure char_cnn.py exists.')
            if not os.path.exists(cnn_model_path):
                raise FileNotFoundError(f'CNN model not found at {cnn_model_path}')
            self.cnn_model = load_cnn_model(cnn_model_path)
            # prepare class mapping: digits 0-9 then A-Z by folder naming assumption used during training
            classes = list(range(10)) + [chr(c) for c in range(ord('A'), ord('Z')+1)]
            self.class_map = [str(x) for x in classes]
        else:
            raise ValueError('Unsupported backend: ' + backend)

    def read_plate(self, plate_img, segments=None):
        """If segments (list of bbox) provided, predict per-segment (for CNN). Otherwise use full-image OCR methods."""
        if self.backend == 'easyocr':
            results = self.reader.readtext(plate_img)
            texts = [r[1] for r in results]
            return ''.join(texts).replace(' ', '')
        elif self.backend == 'tesseract':
            if not _HAS_TESSERACT:
                raise ImportError('pytesseract is not available')
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 7')
            return ''.join(filter(str.isalnum, text))
        elif self.backend == 'cnn':
            if self.cnn_model is None:
                raise RuntimeError('CNN model not loaded')
            # if segments provided, run per-segment, else try to segment using simple method
            if segments is None:
                # fallback: treat whole plate with easyocr as backup
                if _HAS_TESSERACT:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config='--psm 7')
                    return ''.join(filter(str.isalnum, text))
                else:
                    # Use EasyOCR as fallback when Tesseract is not available
                    reader = easyocr.Reader(['en'], gpu=False)
                    results = reader.readtext(plate_img)
                    texts = [r[1] for r in results]
                    return ''.join(texts).replace(' ', '')
            chars = []
            for (x,y,w,h) in segments:
                ch_img = plate_img[y:y+h, x:x+w]
                ch = self._predict_char(ch_img)
                chars.append(ch)
            return ''.join(chars)

    def _predict_char(self, ch_img):
        # preprocess to 28x28 grayscale
        img = cv2.cvtColor(ch_img, cv2.COLOR_BGR2GRAY) if len(ch_img.shape)==3 else ch_img
        img = cv2.resize(img, (28,28))
        img = img.astype('float32')/255.0
        img = img.reshape(1,28,28,1)
        preds = self.cnn_model.predict(img)
        idx = int(preds.argmax(axis=1)[0])
        if idx < len(self.class_map):
            return self.class_map[idx]
        return '?'
