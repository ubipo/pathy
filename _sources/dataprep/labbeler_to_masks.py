"""Converts django-labeller jsons to masks for an entire directory

Images get copied to a rgb/ folder, masks to gt/.

Usage: labbeler_to_masks.py <in_dir> <out_dir>
    in_dir: directory with images and labels from django-labeller
    out_dir: where to put rgb/ and gt/ directories
"""

import sys, os, logging, json, shutil
from pathlib import Path

import cv2

from json2mask.json2mask import json2mask


def labeller_to_masks(labeller_dir: Path, out_dir: Path):
    image_ps = sorted(labeller_dir.glob("*.jpg"))
    json_ps = sorted(labeller_dir.glob("*.json"))

    rgb_dir = out_dir / "rgb"
    gt_dir = out_dir / "gt"
    os.mkdir(rgb_dir)
    os.mkdir(gt_dir)

    for image_p, json_p in zip(image_ps, json_ps):
        with open(json_p) as json_f:
            mask = json2mask(json.load(json_f), image_p)
            stem = image_p.stem
            gt_p = gt_dir / f"{stem}.png"
            cv2.imwrite(str(gt_p.resolve()), mask)
            rgb_p = rgb_dir / f"{stem}.jpg"
            shutil.copy(image_p, rgb_p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    labeller_dir = Path(sys.argv[1])
    assert(labeller_dir.is_dir())

    out_dir = Path(sys.argv[2])
    assert(out_dir.is_dir())
    
    labeller_to_masks(labeller_dir, out_dir)
