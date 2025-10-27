
import numpy as np
import cv2
import os

def compute_homography(image_points, world_points):
    """Compute homography matrix mapping image_points -> world_points.
    image_points: list of [x,y] in image pixels
    world_points: list of [X,Y] in world meters (ground plane)
    Returns 3x3 homography matrix H so that [X,Y,1]^T ~ H * [x,y,1]^T
    """
    if len(image_points) < 4 or len(world_points) < 4:
        # For a planar homography, 4 point correspondences are recommended.
        # If fewer are provided, OpenCV will still attempt but accuracy may be poor.
        raise ValueError('At least 4 correspondences are recommended for robust homography.')
    img_pts = np.array(image_points, dtype=np.float32)
    world_pts = np.array(world_points, dtype=np.float32)
    H, mask = cv2.findHomography(img_pts, world_pts, method=0)
    return H

def load_homography(path):
    data = np.load(path)
    return data['H']

def save_homography(H, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, H=H)
