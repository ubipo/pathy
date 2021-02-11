"""Flatten Giusti et al. frame folder structure

Usage: giusti-flatten.py <in_dir> <out_dir> <metadata_file>

    in_dir: folder with data from https://people.idsia.ch/~guzzi/DataSet.html
    out_dir: the directory to put all frames that will be named like "00072811.jpg"
    metadata_file: file to store correlation between input paths and output frame numbers
"""

import sys, os, json


def main(in_dir_p: str, out_dir_p: str, metadata_p: str):
    frames_dir_ps = []
    for (dirpath, _, _) in os.walk(in_dir_p):
        if dirpath.endswith(".frames"):
            frames_dir_ps.append(dirpath)

    frames_dir_ps.sort()

    frames_dir_counts = []
    frame_counter = 0

    for frame_dir_p in frames_dir_ps:
        _, _, frame_ps = next(os.walk(frame_dir_p))
        frame_ps.sort()
        for frame_p in frame_ps:
            nmbr_str = str(frame_counter).zfill(8)
            frame_name = f"{nmbr_str}.jpg"
            frame_counter += 1
            os.rename(
                os.path.join(frame_dir_p, frame_p),
                os.path.join(out_dir_p, frame_name)
            )
        frames_dir_counts.append(frame_counter)

    metadata = {}
    for (frame_dir_p, count) in zip(frames_dir_ps, frames_dir_counts):
        metadata[count] = frame_dir_p
    
    with open(metadata_p, 'w+') as metadata_f:
        json.dump(metadata, metadata_f)


if __name__ == "__main__":
    in_dir_p = sys.argv[1]
    out_dir_p = sys.argv[2]
    metadata_p = sys.argv[3]
    main(in_dir_p, out_dir_p, metadata_p)
