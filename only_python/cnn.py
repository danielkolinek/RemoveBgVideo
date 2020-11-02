#libs
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from libs.utils import load_graph_model, get_input_tensors, get_output_tensors

# tensorflow not printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

class Resnet:
    def __init__(self, model_path, even):
        self.model = load_graph_model(model_path)
        self.even = even

    def get_mask(self, img):
        imgWidth, imgHeight = img.shape[:2]#img.size

        # Get input and output tensors
        input_tensor_names = get_input_tensors(self.model)
        output_tensor_names = get_output_tensors(self.model)
        input_tensor = self.model.get_tensor_by_name(input_tensor_names[0])

        # Preprocessing Image
        m = np.array([-123.15, -115.90, -103.06])
        x = np.add(img, m)
        sample_image = x[tf.newaxis, ...]
        print("done.\nRunning inference...", end="")

        # evaluate the loaded model directly
        with tf.compat.v1.Session(graph=self.model) as sess:
            results = sess.run(output_tensor_names, feed_dict={
                            input_tensor: sample_image})

        for idx, name in enumerate(output_tensor_names):
            if 'float_segments' in name:
                segments = np.squeeze(results[idx], 0)

        # Segmentation MASk
        segmentation_threshold = 0.7
        mask_small = np.where(segments < segmentation_threshold, [0,0,0], [1,1,1]).astype('uint8')
        mask = cv2.resize(mask_small, (imgHeight,imgWidth))
        return mask