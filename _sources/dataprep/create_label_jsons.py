"""Create empty label json's for django-labeller

Usage: create_label_jsons.py <dir>

    dir: folder with images
"""

import sys, json
from pathlib import Path


def main(directory: Path):
    image_ps = directory.glob('*.jpg')
    for image_p in image_ps:
        json_p = directory / f"{image_p.stem}__labels.json"
        label_obj = {
            "image_filename": image_p.name,
            "completed_tasks": [],
            "labels": []
        }
        with open(json_p, 'w+') as json_f:
            json.dump(label_obj, json_f)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    assert(directory.is_dir())
    main(directory)
