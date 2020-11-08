import cv2
import requests
import numpy as np
import sys
import pyfakewebcam

def load_bg(bg_path, height, width):
    if bg_path != "blur":
        bg = cv2.imread(bg_path)
        return cv2.resize(bg, (width, height))
    else:
        return bg_path

def blur_bg(img):
    #darken bg
    bg_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_hsv[:,:,2] = bg_hsv[...,2]*0.6
    bg_rgb = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)
    #blur bg
    return cv2.blur(bg_rgb,(30,30))
    

def run_video(video_path, bg_path, save_path, frame_counter, get_mask_function, model, threshold):
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
                #if blur, then blur bg
                if bg_path == "blur":
                    bg = blur_bg(frame)
                #get mask
                mask = get_mask_function(frame, bg, model, threshold)
                result = np.where(mask==1, frame, bg) 
                out.write(result)
                #write frame_counter
                sys.stdout.write("\rProcessing video: " + str(round((frame_counter/frames_count)*100))+ "%")
                sys.stdout.flush()
                frame_counter+=1
                #Show frame
                cv2.imshow('frame',result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        print("\nDone processed video saved to: ", save_path)
        video.release()
        out.release()

def run_img(img_path, bg_path, save_path, get_mask_function, model, threshold):
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    bg = load_bg(bg_path, height, width)
    #if blur, then blur bg
    if bg_path == "blur":
        bg = blur_bg(frame)
    mask = get_mask_function(image, bg, model, threshold)
    masked_img = np.where(mask==1, image, bg) 
    cv2.imwrite(save_path, masked_img)

def run_camera(camerain, cameraout, height_setting, width_setting, fps_setting, bg_path, get_mask_function, model, threshold, easteregg):
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
    if easteregg != "starwars":

        while True:
            _, frame = camera.read()
            #if blur, then blur bg
            if bg_path == "blur":
                bg = blur_bg(frame)
            #Show frame
            mask = get_mask_function(frame, bg, model, threshold)
            frame = np.where(mask==1, frame, bg) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fake.schedule_frame(frame)
    else: # if runnig for starwars easteregg
        bg = load_bg("test_inputs/holo_table_chew.jpg", height, width)
        while True:
            _, frame = camera.read()
            #if blur, then blur bg
            #Show frame
            mask = get_mask_function(frame, bg, model, threshold)
            frame = starwars_hologram(frame)

            frame = np.where(mask==1, cv2.addWeighted(bg,0.2,frame,0.8,0), bg) 
            #Show frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fake.schedule_frame(frame)


def starwars_hologram(image):
    height, width = image.shape[:2]
    image = cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)

    #some lines darker
    line = np.random.randint(low=2, high=5)
    image[0::line,:] = image[0::line,:]*np.random.uniform(0.01, 0.2)
    return image