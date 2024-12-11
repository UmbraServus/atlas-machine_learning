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
        with open(classes_path, 'r') as f:
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

    def sigmoid(self, x):
        """sigmoid function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
args:
    outputs: list, np.ndarr containing the pred from the Darknet, single image
    Ea. output will have shape (grid_h, grid_w, anchor_boxes, 4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
    image_size is a np.ndarr containing image's orig. size [image_h, image_w]

Returns: tuple of (boxes, box_confidences, box_class_probs):
    boxes: list, np.ndarr shape (grid_h, grid_w, anchor_boxes, 4)
        4 => (x1, y1, x2, y2)
(x1, y1, x2, y2) should represent the boundary box relative to original image
    box_confidences: list, np.ndarr shape (grid_h, grid_w, anchor_boxes, 1)
    box_class_probs: list, ndarr shape (grid_h, grid_w, anchor_boxes, classes)
    """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size

        for idx, output in enumerate(outputs):
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            grid_h, grid_w = output.shape[:2]
            box_confidence = output[..., 4:5]
            class_prob = output[..., 5:]

            box_conf = self.sigmoid(box_confidence)
            box_class_p = self.sigmoid(class_prob)

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_p)

            i, j = np.indices((grid_h, grid_w))
            i = i[..., np.newaxis]
            j = j[..., np.newaxis]
            grid_x = ((self.sigmoid(t_x)) + j) / grid_w
            grid_y = ((self.sigmoid(t_y)) + i) / grid_h

            anchor_w = self.anchors[idx, :, 0]
            anchor_h = self.anchors[idx, :, 1]
            box_width = np.exp(t_w) * anchor_w / self.model.input.shape[1]
            box_height = np.exp(t_h) * anchor_h / self.model.input.shape[2]

            x_center = grid_x * image_w
            y_center = grid_y * image_h
            x1 = x_center - (box_width * image_w / 2)
            x2 = x_center + (box_width * image_w / 2)
            y1 = y_center - (box_height * image_h / 2)
            y2 = y_center + (box_height * image_h / 2)

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

        return boxes, box_confidences, box_class_probs


    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
args:
boxes: list of np.ndarr of shape (grid_h, grid_w, anchor_boxes, 4) 
containing processed boundary boxes for each output
    
box_confidences: list of np.ndarr of shape (grid_h, grid_w, anchor_boxes, 1) 
containing processed box confidences for each output

box_class_probs: list of np.ndarr of shape (grid_h, grid_w, anchor_boxes, classes) 
containing processed box class probabilities for each output

Returns: (filtered_boxes, box_classes, box_scores):
filtered_boxes: np.ndarr of shape (?, 4) w/ all filtered bounding boxes
box_classes: np.ndarr of shape (?,) w/ class num for each filtered box
box_scores: np.ndarr of shape (?) w/ scores for each filtered box
"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for (box, conf, class_probs) in zip(boxes, box_confidences, box_class_probs):

            box_scores_ = conf * class_probs
            box_classes_ = np.argmax(box_scores_, axis=-1)
            max_scores = np.max(box_scores_, axis=-1)
            filter_mask = max_scores >= self.class_t
            filtered_boxes.append(box[filter_mask])
            box_classes.append(box_classes_[filter_mask])
            box_scores.append(max_scores[filter_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
