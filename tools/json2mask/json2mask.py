""" JSON to mask script

This python script reads a json file (.json) with a fixed format and
converts this to a image mask in PNG format. The mask consits out 2 colors
1: Red, the color of the road 
2: Black, the background

The script accepets 2 parameteres
1: the path to the json file and
2: the path to the original image.

The script outputs the mask in the same folder as the original image.

The script requires OpenCV & NumPy to be installed 
>> pip install opencv-python
>> pip install numpy
"""
import numpy as np
import json
import os
import sys
import cv2

json_path = sys.argv[1]
image_path = sys.argv[2]

f = open(json_path, "r")
im = cv2.imread(image_path)
h,w,c = im.shape
json_dict = json.loads(f.read())
filename = json_dict['image_filename']

mask = []
for region in json_dict['labels'][0]['regions']:

    contours = []
    for coord in region:
        contours.append([ round(coord['x']) , round(coord['y']) ])
    mask.append(contours)

npmask = np.array(mask)

img = np.zeros((h,w))
for npm in npmask:
    cv2.fillPoly(img, pts = [np.array(npm)], color=(255,255,0), lineType=8, shift=0)

cv2.imwrite(os.path.dirname(image_path)+"/" + os.path.splitext(filename)[0] +"_mask.png", img)
