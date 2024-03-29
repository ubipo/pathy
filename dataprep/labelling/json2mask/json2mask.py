""" JSON to mask

Reads a json file (.json) with a fixed format and
converts this to a grayscale png mask. With:
- White (#FF): as the color of the road 
- Black (#00): as the background

Usage: `json2mask.py <json_file> <rgb_image>`

    json_file: path to the json file
    rgb_image: path to the rgb image

The script outputs the mask as `<image_stem>_mask.png` to the same folder as the original image.

Requires OpenCV & NumPy:
>> pip install opencv-python
>> pip install numpy
"""
from pathlib import Path
import numpy as np
import json
import os
import sys
import cv2

def json2mask(json: object, image_path: Path):
    im = cv2.imread(str(image_path.resolve()))
    h, w, _ = im.shape

    labels = json['labels']
    polys = []
    for label in labels:
        for region in label['regions']:
            poly = []
            for coord in region:
                poly.append([round(coord['x']), round(coord['y'])])
            polys.append(poly)

    mask = np.zeros((h, w))
    for poly in polys:
        cv2.fillPoly(mask, pts = [np.array(poly)], color=[255,255,255], lineType=8, shift=0)

    return mask

def jsonfile2maskfile(json_path: Path, image_path: Path, mask_out_path: Path):
    with open(json_path) as json_f:
        mask = json2mask(json.load(json_f), image_path)
        cv2.imwrite(str(mask_out_path.resolve()), mask)


if __name__ == "__main__":
    json_path = Path(sys.argv[1])
    assert(json_path.is_file())
    image_path = Path(sys.argv[2])
    assert(image_path.is_file())

    mask_out_path = Path(f"{image_path.stem}_mask.png")
    jsonfile2maskfile(json_path, image_path, mask_out_path)
