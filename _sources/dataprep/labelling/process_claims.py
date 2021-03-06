"""Process images from django-labeller and a claims csv

Iterates through every claim, checks json and comment, and
copies both the corresponding json file and image to <out_dir>.

Usage: `process_labelled_images.py <in_dir> <claims_csv> <out_dir>`

    in_dir: directory with images and labels from django-labeller
    claims_csv: csv from the Google sheet we used
    out_dir: where to put processed json and images

tfr files get written to a new tfr/ directory within <dir>
"""

import sys, os, csv, logging, json, shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np

from json2mask.json2mask import json2mask


CLAIMED_BY = "Claimed by"
IMAGE_NBR = "image #"
COMMENTS = "Comments"


def image_ps_from_directory(directory: Path) -> List[str]:
    """**Must** be the same as image path resoltion code
    used with django-labeller, otherwise image# will be
    mismatched
    """
    ps_from_dir = lambda dir: [str(p.resolve()) for p in dir.glob("*.jpg")]
    image_paths_l = ps_from_dir((directory / "l"))
    image_paths_l.sort()
    image_paths_r = ps_from_dir((directory / "r"))
    image_paths_r.sort()
    image_paths_s = ps_from_dir((directory / "s"))
    image_paths_s.sort()
    return image_paths_s + image_paths_l + image_paths_r

def get_labelled_images(directory: Path, claims_p: Path, out_dir: Path):
    image_ps = image_ps_from_directory(directory)

    nbro_processed_files = 0

    with open(claims_p, newline='') as claims_csv:
        claims_r = csv.DictReader(claims_csv)
        for claim in claims_r:
            claimed_by = claim[CLAIMED_BY]
            if claimed_by == "":
                continue
            image_nbr = int(claim[IMAGE_NBR])
            comments = claim[COMMENTS]
            if comments != "":
                logging.warning(f"Comment for #{image_nbr}: \"{comments}\" (claimed: \"{claimed_by}\")")
            
            image_p = Path(image_ps[image_nbr - 1]) # Django labeller starts at 1
            json_p = image_p.parent / f"{image_p.stem}__labels.json"
            
            label_json = None
            try:
                with open(json_p) as json_f:
                    label_json = json.load(json_f)
            except FileNotFoundError:
                logging.warning(f"No json file found for #{image_nbr}, assuming empty (claimed: \"{claimed_by}\", image_p: {image_p})")
                label_json = {
                    "image_filename": image_p.stem,
                    "completed_tasks": [],
                    "labels": []
                }

            mask_image = json2mask(label_json, image_p)
            nbro_non_zero_pixels =  np.count_nonzero(mask_image)
            if nbro_non_zero_pixels == 0:
                logging.warning(f"No mask drawn for #{image_nbr} (claimed: \"{claimed_by}\", image_p: {image_p})")

            json_out_p = out_dir / f"{image_p.stem}__labels.json"
            with open(json_out_p, 'w+') as json_out_f:
                json.dump(label_json, json_out_f)
            copied_image_p = out_dir / image_p.name
            shutil.copy(image_p, copied_image_p)

            nbro_processed_files += 1

    logging.info(f"Processed {nbro_processed_files} masks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    directory = Path(sys.argv[1])
    assert(directory.is_dir())

    claims_p = Path(sys.argv[2])
    assert(claims_p.is_file())

    out_dir = Path(sys.argv[3])
    assert(out_dir.is_dir())
    
    get_labelled_images(directory, claims_p, out_dir)
