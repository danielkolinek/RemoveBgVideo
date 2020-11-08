#libs
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from only_python.libs.utils import load_graph_model, get_input_tensors, get_output_tensors

# tensorflow not printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

class Resnet:
    def __init__(self, model_path, even):
        self.model = load_graph_model(model_path)
        self.even = even
        # get input and output tensors
        input_tensor_names = get_input_tensors(self.model)
        self.output_tensor_names = get_output_tensors(self.model)
        self.input_tensor = self.model.get_tensor_by_name(input_tensor_names[0])

    def get_mask(self, img, bg, _, threshold):
        imgWidth, imgHeight = img.shape[:2]

        # Preprocessing Image
        # add imagenet mean - extracted from body-pix source
        mean = np.array([-123.15, -115.90, -103.06])
        x = np.add(img, mean)
        sample_image = x[tf.newaxis, ...]

        # evaluate the loaded model directly
        with tf.compat.v1.Session(graph=self.model) as sess:
            results = sess.run(self.output_tensor_names, feed_dict={
                            self.input_tensor: sample_image})

        for idx, name in enumerate(self.output_tensor_names):
            if 'float_segments' in name:
                segments = np.squeeze(results[idx], 0)

        # segmentation mask
        mask_small = np.where(segments < threshold, [0,0,0], [1,1,1]).astype('uint8')
        return cv2.resize(mask_small, (imgHeight,imgWidth)) 