"""Flatten Yamaha folder structure

Usage: `yamaha_flatten.py <in_dir> <out_dir>`

    in_dir: folder with data from https://people.idsia.ch/~guzzi/DataSet.html
    out_dir: the directory to put all frames that will be named like "00072811.jpg"
"""

import sys, os, json


def main(in_dir_p: str, out_dir_p: str):
    frames_dir_ps = []
    for (dirpath, _, _) in os.walk(in_dir_p):
        if os.path.split(dirpath)[-1].startswith("iid"):
            frames_dir_ps.append(dirpath)

    frames_dir_ps.sort()
    frame_counter = 0

    for frame_dir_p in frames_dir_ps:
        nmbr_str = str(frame_counter).zfill(8)
        frame_name = f"{nmbr_str}.jpg"
        labels_name = f"{nmbr_str}.labels.png"
        frame_counter += 1
        os.rename(
            os.path.join(frame_dir_p, "rgb.jpg"),
            os.path.join(out_dir_p, frame_name)
        )
        os.rename(
            os.path.join(frame_dir_p, "labels.png"),
            os.path.join(out_dir_p, labels_name)
        )


if __name__ == "__main__":
    in_dir_p = sys.argv[1]
    out_dir_p = sys.argv[2]
    main(in_dir_p, out_dir_p)
