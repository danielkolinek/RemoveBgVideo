import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import argparse
import cv2
import numpy as np

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

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#load model
bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

#load bg
bg = cv2.imread(args.bg)

def get_masked_img(img, bg, bodypix_model):
    image_array = tf.keras.preprocessing.image.img_to_array(frame)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.75)
    return np.where(mask==1, frame, bg) 


if args.video is not None:
    video = cv2.VideoCapture(args.video)
    if video.isOpened(): 
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.save, int(cv2.CAP_FFMPEG),int(cv2.VideoWriter_fourcc('M','J','P','G')), fps, (width,height))
        #resize bg
        bg = cv2.resize(bg, (width, height))
        while(video.isOpened()):
            ret, frame = video.read()
            if ret==True:
                #get mask
                result = get_masked_img(frame, bg, bodypix_model)
                out.write(result)
                print("OK")
            else:
                break
    video.release()
    out.release()
else:
    cam = cv2.VideoCapture('/dev/video0')
    height, width = 720, 1280
    bg = cv2.resize(bg, (width, height))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 60)
    while True:
        _, frame = cam.read()
        res = get_masked_img(frame, bg, bodypix_model)
        cv2.imshow('frame',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()


