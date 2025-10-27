
import os, cv2
from src.character_segmentation import segment_characters

def extract_chars_from_plate(plate_img, out_dir, prefix='plate'):
    os.makedirs(out_dir, exist_ok=True)
    boxes = segment_characters(plate_img)
    for i, (x,y,w,h) in enumerate(boxes):
        ch = plate_img[y:y+h, x:x+w]
        out_path = os.path.join(out_dir, f"{prefix}_ch_{i}.png")
        cv2.imwrite(out_path, ch)
    return len(boxes)
