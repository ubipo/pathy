"""Remove sequential and similar images

Usage: remove_seq_sim.py <dir> <threshold>

    dir: folder with images
    threshold: accumulated Chi-square histogram distance cutoff for unique images, higher is less similar

"sequential" means sorted()
See also: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

Deletion ~percentage by threshold values for giusti:
32: 47
35: 49
50: 56
70: 63
80: 67
150: 78
200: 81
300: 86 <-
400: 88
"""

import os, sys, collections, statistics, shutil

import cv2 as cv
import progressbar
import click


CHISQUARE = 1
# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]


def main(dir_p: str, threshold: float):
    _, _, image_names = next(os.walk(dir_p))
    image_names.sort()
    image_ps = [os.path.join(dir_p, image_name) for image_name in image_names]
    
    image_ps_to_delete = []
    last_hists = collections.deque([], 3)
    with progressbar.ProgressBar(max_value=len(image_ps)) as bar:
        for i, image_p in enumerate(image_ps):
            image = cv.imread(image_p)
            image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            hist = cv.calcHist([image_hsv], channels, None, histSize, ranges, accumulate=False)
            cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

            if len(last_hists) > 1:
                dists = [cv.compareHist(last_hist, hist, CHISQUARE) for last_hist in last_hists]
                dists_avg = statistics.mean(dists)
                if dists_avg < threshold:
                    image_ps_to_delete.append(image_p)

            last_hists.appendleft(hist)
            bar.update(i)

    image_ps_to_keep = [image_p for image_p in image_ps if image_p not in image_ps_to_delete]

    images_len = len(image_names)
    images_to_delete_len = len(image_ps_to_delete)
    deleted_percentage = round((images_to_delete_len / images_len) * 100, 2)
    kept_percentage = 100 - deleted_percentage
    print(f"Deleting {images_to_delete_len} of {images_len} frames (delete {deleted_percentage}%, keep {kept_percentage}%)")

    if click.confirm('Copy kept images to "kept" subdir'):
        kept_dir_p = os.path.join(dir_p, "kept")
        os.mkdir(kept_dir_p)
        for image_p in image_ps_to_keep:
            shutil.copy(image_p, kept_dir_p)


if __name__ == "__main__":
    dir_p = sys.argv[1]
    threshold = float(sys.argv[2])
    main(dir_p, threshold)
