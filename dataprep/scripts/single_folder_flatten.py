"""Flatten Yamaha folder structure

Usage: `yamaha_flatten.py <in_dir> <out_dir>`

    in_dir: folder with data from https://people.idsia.ch/~guzzi/DataSet.html
    out_dir: the directory to put all frames that will be named like "00072811.jpg"
"""

import sys, os
from pathlib import Path


def main(in_dir_p: Path, out_dir_p: Path):
    frame_ps = sorted(in_dir_p.glob("*.jpg"))

    for i, frame_p in enumerate(frame_ps):
        nmbr_str = str(i).zfill(8)
        frame_name = f"{nmbr_str}.jpg"
        os.rename(frame_p, out_dir_p / frame_name)


if __name__ == "__main__":
    in_dir_p = Path(sys.argv[1])
    assert(in_dir_p.is_dir())
    out_dir_p = Path(sys.argv[2])
    assert(out_dir_p.is_dir())
    main(in_dir_p, out_dir_p)
