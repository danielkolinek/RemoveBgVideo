# Remove background with bodypix
This project contains 3 ways to implement backgroung remove from image, when foreground is person's body. 

## 1. Python + nodejs
This version was created becouse of being able to run on GPU. In time of release there is no other way to use tensorflowjs on gpu with python then sending requests to server. Tensorflowjs in python is runned from tensorflowjs in tensorflow not tensorflow-gpu ..

### Requirements
- Python 3
    - numpy, opencv, argparse, requets, pyfakewebcam
- Nodejs 
    - tensorflow-models/body-pix, tensorflow/tfjs-node, tensorflow/tfjs-node-gpu
    - all for node should be installed just by going to folder node_modules and running `npm install`, where everything will be installed from package.json
- Nvidia
    - installed and running nvidia driver for your card
    - cuda 10 (I have tried cuda 10 for ubuntu 18.04 on ubuntu 20.04. Everything is running for this implementation, but is not faster then bodypix-api, so I do not recomend)
    - in case you do not have nvidia or do not want to use this on gpu, just rewrite `require('@tensorflow/tfjs-node-gpu')` to `require('@tensorflow/tfjs-node')` in file in node_modules/app.json.
- Fake camera
    - v4l2loopback-dkms, modprobe

### Before running script
- run server
    - `nodejs node_part/app.js 0.75` 
    - the 0.75 is 
- for camera firstly fake camera has to be created (run as sudo)
    - `modprobe -r v4l2loopback`
    - `sudo modprobe v4l2loopback devices=1 video_nr=5 card_label="v4l2loopback" exclusive_caps=1`

## 2. Bodypix-api
This is maybe the best way for now. Everything needed is here: https://pypi.org/project/tf-bodypix/

### Requirements
- Python 3
    - numpy, opencv, argparse, pyfakewebcam, tf_bodypix, tensorflow
- Fake camera
    - v4l2loopback-dkms, modprobe

### Before running script
- for camera firstly fake camera has to be created (run as sudo). Same like before.

## 3. Only python
Version using only python, tensorflow, opencv and numpy.

Special thanks to https://github.com/ajaichemmanam/simple_bodypix_python

Because of running only on cpu and not very optimal implementation, this is too slow for live camera
and I do not recomend using this on longer videos (20s+) or videos with higher resolution (hd+).

### Requirements
- Python 3
    - numpy, opencv, argparse, tensorflow

## Before running script

Download resnet model with script:
- `bash download_model.sh`

## Tested and running on:

- Python: Python 3.8.5, tensorflow 2.3.1, opencv-python 4.3.0.36, numpy 1.18.5, tf-bodypix 0.3.1
- OS: Ubuntu 20.04
- Nvidia: NVIDIA-SMI 455.32.00, Cuda 10

## Run script
`remove_bg.py [-h] [--o option] [--img /path/to/image] [--video /path/to/video] --bg /path/to/bg [--save filename] [--camerain videoIn] [--cameraout videoOut] [--height heightForCamera] [--width widthForCamera] [--fps fpsForCamera] [--threshold thresholdForDetection] [--starwars] `

- `-h` 
    - for help
- `--o`
    - Option of version you want to run\
        1 = python+node\
        2 = bodypix-api\
        3 = only_python 
- `--img` / `--video`
    - for input image or video, choose only one, if you wanna use video, be prepared, couse it takes a lot of time.
- `--bg`
    - backgroung that will replace oldone  
    - for blur bg use `blur`
- `--save`
    - filename for result
- `--threshold`
    - threshold for cnn
- `--starwars`
    - Easter egg holo

Setings for camera:
- `--camerain`
    - camera input for bg delete (for example /dev/video0)
- `--cameraout`
    - fake camera for output (for example /dev/video5)
- `--height` 
    - optional (default is 640), height for camera
- `--width` 
    - optional (default is 480), width for camera
- `--fps` 
    - optional (default is 60), fps for camera

Example: 
    `python3 remove_bg.py --o 2 --save result.mp4 --bg test_inputs/bg.jpg --video test_inputs/test.mp4`