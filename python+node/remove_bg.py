#libs
import requests
import sys
import argparse
import numpy as np
import cv2
import argparse

"""
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
"""

def get_mask(frame, bodypix_url='http://127.0.0.1'):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    # convert raw bytes to a numpy array
    # raw data is uint8[width * height] with value 0 or 1
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask

cap = cv2.VideoCapture('/dev/video0')
success, frame = cap.read()
cv2.imwrite("result",get_mask(frame))