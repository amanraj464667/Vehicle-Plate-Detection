
import cv2
import imutils
import numpy as np

def detect_plate_contour(image):
    """Baseline plate localization using OpenCV contour approximation.
    Returns bounding box (x,y,w,h) of detected plate or None.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(image_gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 170, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

    plate_box = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            # heuristic checks for aspect ratio and size
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6 and w > 60 and h > 15:
                plate_box = (x, y, w, h)
                break
    return plate_box

def crop_plate(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w].copy()
