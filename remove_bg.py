#libs
import requests
import sys
import argparse
import numpy as np
import cv2
import argparse
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
#custom imports
from functions.functions import run_video, run_img, run_camera
from only_python.cnn import Resnet

def get_masked_img_python_node(image, bg, _):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, data = cv2.imencode(".jpg", image_rgb)
    r = requests.post(
        url='http://127.0.0.1:8080',
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    # convert raw bytes to a numpy array
    # raw data is uint8[width * height] with value 0 or 1
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((image.shape[0], image.shape[1]))
    # reshape mask from 2d to 3d
    mask = np.repeat(np.expand_dims(mask, axis=2), 3,  axis=2)
    return np.where(mask==1, image, bg)

def get_masked_img_bodypix_api(image, bg, bodypix_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = bodypix_model.predict_single(image_rgb)
    mask = result.get_mask(threshold=0.5)
    return np.where(mask==1, image, bg)

# run like python3 morph.py --image_1 ../img/DB1_B/101_2.tif --image_2 ../img/DB1_B/102_2.tif --blocksize 10
parser = argparse.ArgumentParser(
    description='Remove background from image or video using only python')
parser.add_argument("--o",
                    metavar="option", required=True,
                    help="Option of version you want to run\n\
                        1 = python+node\n\
                        2 = bodypix-api\n\
                        3 = only_python")
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

def get_right_mask_function(option):
    if option == "1":
        return get_masked_img_python_node, ""
    elif option == "2":
        bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        return get_masked_img_bodypix_api, bodypix_model
    elif option == "3":
        resnet_model = Resnet('only_python/json_model/model.json', 16)
        return resnet_model.get_mask, ""


get_masked_img_function, model = get_right_mask_function(args.o)
#if input is video
if args.video is not None and args.bg is not None and args.save is not None:
    frame_counter = 0
    run_video(args.video, args.bg, args.save, frame_counter, get_masked_img_function, model)
#if input is video
elif args.img is not None and args.bg is not None and args.save is not None:
    run_img(args.img, args.bg, args.save, get_masked_img_function, model)
#if input is camera and want to produce fake video
elif args.camerain is not None and args.cameraout is not None:
    if args.o == "3":
        print("Sorry but for this use option 3 can not be used")
        exit()
    run_camera(args.camerain, args.cameraout, args.height, args.width, args.fps, args.bg, get_masked_img_function, model)

#mistake
else:
    parser.print_help()

