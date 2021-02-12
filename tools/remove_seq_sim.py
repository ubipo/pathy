"""Remove sequential and similar images

Usage: remove_seq_sim.py <dir> <threshold>

    dir: folder with images
    threshold: accumulated Chi-square histogram distance cutoff for unique images, higher is less similar

"sequential" means sorted()
See also: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

Deletion percentage by threshold values for giusti:
32: 47.73%
35: 49.42%
50: 56.83%
70: 63.58%
"""

import os, sys, collections, statistics

import cv2 as cv
import progressbar


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
    
    image_ps_to_delete = []
    last_hists = collections.deque([], 3)
    with progressbar.ProgressBar(max_value=len(image_names)) as bar:
        for i, image_name in enumerate(image_names):
            image_p = os.path.join(dir_p, image_name)
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
        
            if i == 170:
                break

    print(image_ps_to_delete)

    images_len = len(image_names)
    images_to_delete_len = len(image_ps_to_delete)
    deleted_percentage = round((images_to_delete_len / images_len) * 100, 2)
    kept_percentage = 100 - deleted_percentage
    print(f"Deleting {images_to_delete_len} of {images_len} frames (delete {deleted_percentage}%, keep {kept_percentage}%)")


if __name__ == "__main__":
    dir_p = sys.argv[1]
    threshold = float(sys.argv[2])
    main(dir_p, threshold)
