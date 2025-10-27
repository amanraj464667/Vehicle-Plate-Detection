
import cv2
import numpy as np

def show_image(window_name, img, scale=1.0):
    # Utility to display an image (for local testing)
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(window_name, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
