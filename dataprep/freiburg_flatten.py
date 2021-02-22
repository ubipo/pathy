"""Flatten Freiburg folder structure

Usage: freiburg_flatten.py <in_dir> <out_dir>

    in_dir: "train" folder from ... TODO: url
    out_dir: the directory to put all frames that will be named like "00072811.jpg"
"""

import sys, os, json


def main(in_dir_p: str, out_dir_p: str):
    rgb_folder_p = os.path.join(in_dir_p, "rgb")
    gt_folder_p = os.path.join(in_dir_p, "GT_color")

    _, _, rgb_frame_names = next(os.walk(rgb_folder_p))
    _, _, gt_frame_names = next(os.walk(gt_folder_p))

    assert(len(rgb_frame_names) == len(gt_frame_names))

    rgb_frame_names.sort()
    gt_frame_names.sort()

    frame_counter = 0
    for (rgb_frame_p, gt_frame_p) in zip(rgb_frame_names, gt_frame_names):
        nmbr_str = str(frame_counter).zfill(8)
        frame_name = f"{nmbr_str}.jpg"
        gt_name = f"{nmbr_str}.gt-f.png"
        frame_counter += 1
        os.rename(
            os.path.join(rgb_folder_p, rgb_frame_p),
            os.path.join(out_dir_p, frame_name)
        )
        os.rename(
            os.path.join(gt_folder_p, gt_frame_p),
            os.path.join(out_dir_p, gt_name)
        )


if __name__ == "__main__":
    in_dir_p = sys.argv[1]
    out_dir_p = sys.argv[2]
    main(in_dir_p, out_dir_p)
