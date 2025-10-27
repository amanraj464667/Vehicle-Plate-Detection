import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

DEFAULT_URLS = [
    # Fill this with a valid license-plate YOLOv8 weights URL or provide your own with --url
]

def main():
    parser = argparse.ArgumentParser(description='Download YOLOv8 license-plate weights')
    parser.add_argument('--url', default=None, help='Direct URL to weights .pt file')
    parser.add_argument('--out', default=str(Path('models') / 'yolov8n-license-plate.pt'), help='Destination path')
    args = parser.parse_args()

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = args.url or (DEFAULT_URLS[0] if DEFAULT_URLS else None)
    if url is None:
        print('No URL provided. Please re-run with --url <weights_url> (a YOLOv8 license-plate .pt file).')
        print(f'Destination path will be: {dest}')
        sys.exit(1)

    if requests is None:
        print('requests is not installed. pip install requests, or download manually.')
        sys.exit(1)

    print(f'Downloading weights from {url} -> {dest}')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print('Done.')

if __name__ == '__main__':
    main()
