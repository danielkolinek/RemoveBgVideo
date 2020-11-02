#libs
import sys
import argparse
import numpy as np
import cv2
import argparse



# run like python3 morph.py --image_1 ../img/DB1_B/101_2.tif --image_2 ../img/DB1_B/102_2.tif --blocksize 10
parser = argparse.ArgumentParser(
    description='Remove background from image or video using only python')
parser.add_argument("--img",
                    metavar="/path/to/image", required=False,
                    help="Path to image which is containing bg that has to be removed.")
parser.add_argument("--video",
                    metavar="/path/to/video", required=False,
                    help="Path to video which is containing bg that has to be removed.")
parser.add_argument('--bg', required=True,
                    metavar="/path/to/bg",
                    help="Backgroung that will replace oldone")
parser.add_argument('--save', required=True,
                    metavar="filename",
                    help="filename for result (example \"--save result.jpg\" will create result.jpg)")
args = parser.parse_args()

