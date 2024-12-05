#!/usr/bin/env python3
""" module for Class Yolo that uses YOLOv3 to perform obj detection"""
import numpy as np
import tensorflow as tf
from keras import models
class Yolo():
    """ class for using Yolov3 to perform obj detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
    args:
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to list of class names used for the Darknet
        class_t: float, box score threshold for the initial filtering step
        nms_t: float representing the IOU threshold for non-max suppress.
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2) 
        containing all of the anchor boxes
            outputs: number of outputs (predictions) made by the Darknet model
            anchor_boxes: number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height] """

        self.model = models.load_model(model_path)
        with open(classes_path,'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        if not isinstance(class_t, float):
            raise TypeError("class_t must be a float")
        self.class_t = class_t
        if not isinstance(nms_t, float):
            raise TypeError("nms_t must be a float")
        self.nms_t = nms_t
        if not isinstance(anchors, np.ndarray):
            raise TypeError("anchors must be a np.ndarray")
        self.anchors = anchors
