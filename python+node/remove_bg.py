#libs
import requests
import sys
import argparse
import numpy as np
import cv2
import argparse
#custom imports
from functions import run_video, run_img, run_camera

# run like python3 morph.py --image_1 ../img/DB1_B/101_2.tif --image_2 ../img/DB1_B/102_2.tif --blocksize 10
parser = argparse.ArgumentParser(
    description='Remove background from image or video using only python')
parser.add_argument("--img",
                    metavar="/path/to/image", required=False,
                    help="Path to image which is containing bg that has to be removed.")
parser.add_argument("--video",
                    metavar="/path/to/video", required=False,
                    help="Path to video which is containing bg that has to be removed.")
parser.add_argument('--camerain', required=False,
                    metavar="input device",
                    help="input device, for example: /dev/video0")
parser.add_argument('--cameraout', required=False,
                    metavar="output device",
                    help="output device, for example: /dev/video1")
parser.add_argument("--width",
                    metavar="camera width", required=False,
                    help="For live camera input.")
parser.add_argument("--height",
                    metavar="camera height", required=False,
                    help="For live camera input.")
parser.add_argument("--fps",
                    metavar="camera fps", required=False,
                    help="For live camera input.")
parser.add_argument('--bg', required=True,
                    metavar="/path/to/bg",
                    help="Backgroung that will replace oldone")
parser.add_argument('--save', required=False,
                    metavar="filename",
                    help="filename for result (example \"--save result.jpg\" will create result.jpg)")
args = parser.parse_args()



#if input is image
if args.video is not None and args.bg is not None and args.save is not None:
    frame_counter = 0
    run_video(args.video, args.bg, args.save, frame_counter)

elif args.img is not None and args.bg is not None and args.save is not None:
    run_img(args.img, args.bg, args.save)

elif args.camerain is not None and args.cameraout is not None:
    run_camera(args.camerain, args.cameraout, args.height, args.width, args.fps, args.bg)

else:
    parser.print_help()

"""
cap = cv2.VideoCapture('/dev/video0')
success, frame = cap.read()
cv2.imshow("result.png",get_mask(frame)*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""