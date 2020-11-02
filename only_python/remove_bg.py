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

#init model
from cnn import Resnet
resnet = Resnet('json_model/model.json', 16)

#load bg
bg = cv2.imread(args.bg)

# load input img 
if args.img is not None:
    img = cv2.imread(args.img)
    height, width = img.shape[:2]
    #resize bg
    bg = cv2.resize(bg, (width, height))
    #get mask
    mask = resnet.get_mask(img)
    result = np.where(mask==1, img, bg) 
    cv2.imwrite(args.save, result)
# load video 
elif args.video is not None:
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
                mask = resnet.get_mask(frame)
                result = np.where(mask==1, frame, bg) 
                out.write(result)
            else:
                break
        video.release()
        out.release()
else:
    parser.print_help(sys.stderr)
    exit(1)



