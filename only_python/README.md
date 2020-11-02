# Delete bg from image / video

Version using only python, tensorflow, opencv and numpy.

Special thanks to https://github.com/ajaichemmanam/simple_bodypix_python

## Tested and running on:
Python 3.8.5, tensorflow 2.3.1, opencv-python 4.3.0.36, numpy 1.18.5

## Before run

Download resnet model with script:
- `bash download_model.sh`

## Run
`remove_bg.py [-h] [--img /path/to/image] [--video /path/to/video] --bg /path/to/bg --save filename`

- `-h` 
    - for help
- `--img` / `--video`
    - for input image or video, choose only one, if you wanna use video, be prepared, couse it takes a lot of time.
- `--bg`
    - backgroung that will replace oldone  
- `--save`
    - filename for result