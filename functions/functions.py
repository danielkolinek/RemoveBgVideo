import cv2
import requests
import numpy as np
import sys
import pyfakewebcam

def load_bg(bg_path, height, width):
    bg = cv2.imread(bg_path)
    return cv2.resize(bg, (width, height))

def run_video(video_path, bg_path, save_path, frame_counter, get_masked_img_function, model):
    #open video and get info
    video = cv2.VideoCapture(video_path)
    if video.isOpened(): 
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #load bg
        bg = load_bg(bg_path, height, width)
        #create videowriter
        out = cv2.VideoWriter(save_path, int(cv2.CAP_FFMPEG),int(cv2.VideoWriter_fourcc(*'MP4V')), fps, (width,height))
        #run alg on video and write result to out
        while(video.isOpened()):
            ret, frame = video.read()
            if ret==True:
                #get mask
                result = get_masked_img_function(frame, bg, model)
                out.write(result)
                #write frame_counter
                sys.stdout.write("\rProcessing video: " + str(round((frame_counter/frames_count)*100))+ "%")
                sys.stdout.flush()
                frame_counter+=1
            else:
                break
        print("\nDone processed video saved to: ", save_path)
        video.release()
        out.release()

def run_img(img_path, bg_path, save_path, get_masked_img_function, model):
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    bg = load_bg(bg_path, height, width)
    masked_img = get_masked_img_function(image, bg, model)
    cv2.imwrite(save_path, masked_img)

def run_camera(camerain, cameraout, height_setting, width_setting, fps_setting, bg_path, get_masked_img_function, model):
    #get params
    height = int(height_setting) if height_setting is not None else 480
    width = int(width_setting) if width_setting is not None else 640
    fps = int(fps_setting) if fps_setting is not None else 60
    #set camera
    camera = cv2.VideoCapture(camerain)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FPS, fps)
    #get bg and resize it
    bg = load_bg(bg_path, height, width)
    # setup fake cam
    fake = pyfakewebcam.FakeWebcam(cameraout, width, height)
    # run camera forever
    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(get_masked_img_function(frame, bg, model), cv2.COLOR_BGR2RGB)
        fake.schedule_frame(frame)

