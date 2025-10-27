
import argparse, json, os, numpy as np
from src.homography import compute_homography, save_homography

def run_calibrate(json_path, out_path='models/homography.npz'):
    if not os.path.exists(json_path):
        print('JSON file with correspondences not found:', json_path)
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_pts = data.get('image_points', None)
    world_pts = data.get('world_points', None)
    if img_pts is None or world_pts is None:
        print('JSON must contain "image_points" and "world_points" arrays.')
        return
    H = compute_homography(img_pts, world_pts)
    save_homography(H, out_path)
    print('Saved homography matrix to', out_path)
    print('Homography matrix:\n', H)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSON file with correspondences')
    parser.add_argument('--out', default='models/homography.npz', help='Output NPZ path to save H')
    args = parser.parse_args()
    run_calibrate(args.input, args.out)
