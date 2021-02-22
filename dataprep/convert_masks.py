"""Convert multicolor masks (yamaha/freiburg) to our style (b/w)

Usage: convert_yamaha_masks.py <dir> <color_scheme>

    dir: folder with images
    color_scheme: 0 for yamaha, 1 for freiburg
"""

import os, sys, collections, statistics

import cv2 as cv
import numpy as np
import progressbar


SCHEME_YAMAHA = 0
SCHEME_FREIBURG = 1

# Yamaha's scheme
Y_SMOOTH_TRAIL = [153, 176, 178] # grey
Y_ROUGH_TRAIL = [30, 76, 156] # brown

# Freiburg's scheme
F_ROAD = [170, 170, 170] # grey


def main(dir_p: str, color_scheme: int):
    _, _, image_names = next(os.walk(dir_p))
    image_names.sort()

    mask_condition = None
    if color_scheme == SCHEME_YAMAHA:
        mask_condition = lambda image: (image==Y_SMOOTH_TRAIL) | (image==Y_ROUGH_TRAIL)
    elif color_scheme == SCHEME_FREIBURG:
        mask_condition = lambda image: (image==F_ROAD)
    else:
        assert(False)

    with progressbar.ProgressBar(max_value=len(image_names)) as bar:
        for i, image_name in enumerate(image_names):
            if not (image_name.endswith(".gt-f.png") or image_name.endswith(".gt-y.png")):
                continue
            image_p = os.path.join(dir_p, image_name)
            image = cv.imread(image_p)
            path_mask = np.where(mask_condition(image).all(axis=2))
            h, w, _ = image.shape
            converted = np.zeros((h,w))
            converted[path_mask] = [255]
            bar.update(i)
            extensionless = image_name.split(".")[0]
            converted_p = os.path.join(dir_p, extensionless + ".gt.png")
            cv.imwrite(converted_p, converted) 


if __name__ == "__main__":
    dir_p = sys.argv[1]
    color_scheme = int(sys.argv[2])
    main(dir_p, color_scheme)
