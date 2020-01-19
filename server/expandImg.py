# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os
import glob
img_dir = "" # Enter Directory of all images 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True,
	help="path to the image file")
args = vars(ap.parse_args())

# load the image from disk
img_dir = args["src"]
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
itemNum = len(files)
for f1 in files:
    if "reverse" in f1 or "bright" in f1:
        continue
    img = cv2.imread(f1)
    flipHorizontal = cv2.flip(img, 1)
    #cv2.imshow('Flipped horizontal image', flipHorizontal)
    name = f1[:-4] + "reverse.jpg"
    cv2.imwrite(name, flipHorizontal)
    value = 100

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    name2 = name[:-4] + "bright1.jpg"
    cv2.imwrite(name2, img2)

    lim = 0 + value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img3 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    name3 = name[:-4] + "bright2.jpg"
    cv2.imwrite(name3, img3)
    
    cv2.imshow('Flipped horizontal image', img)
    cv2.waitKey(0)
    cv2.imshow('Flipped horizontal image', img2)
    cv2.waitKey(0)
    cv2.imshow('Flipped horizontal image', img3)
    cv2.waitKey(0)

    