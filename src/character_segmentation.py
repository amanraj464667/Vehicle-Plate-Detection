
import cv2
import numpy as np

def segment_characters(plate_img):
    """Simple character segmentation using vertical projections / contours.
    Returns list of (x,y,w,h) bounding boxes relative to plate image.
    This is a baseline and might need heuristics/tuning for Indian plates.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert so characters are white
    thresh = 255 - thresh
    # find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h > 0.4 * plate_img.shape[0] and w > 5:
            boxes.append((x,y,w,h))
    # sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes
