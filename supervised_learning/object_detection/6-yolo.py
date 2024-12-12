#!/usr/bin/env python3
""" module for Class Yolo that uses YOLOv3 to perform obj detection"""
import numpy as np
import tensorflow as tf
from keras import models
import os
import cv2


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

box_class_probs: list np.ndarr, shape (grid_h, grid_w, anchor_boxes, classes)
containing processed box class probabilities for each output

Returns: (filtered_boxes, box_classes, box_scores):
filtered_boxes: np.ndarr of shape (?, 4) w/ all filtered bounding boxes
box_classes: np.ndarr of shape (?,) w/ class num for each filtered box
box_scores: np.ndarr of shape (?) w/ scores for each filtered box
"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for (b, conf, clsP) in zip(boxes, box_confidences, box_class_probs):

            box_scores_ = conf * clsP
            box_classes_ = np.argmax(box_scores_, axis=-1)
            max_scores = np.max(box_scores_, axis=-1)
            filter_mask = max_scores >= self.class_t
            filtered_boxes.append(b[filter_mask])
            box_classes.append(box_classes_[filter_mask])
            box_scores.append(max_scores[filter_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
args:

filtered_boxes: np.ndarr of shape (?, 4) containin all filtered boundin bxes

box_classes: np.ndarr of shape (?,) containing class num for filtered_boxes

box_scores: np.ndarr of shape (?) containing scores for each filtered box

Returns tuple (box_predictions, predicted_box_classes, predicted_box_scores):
    box_predictions: np.arr of shape (?, 4) containin predicted boundin bxes
    ordered by class and score

    predicted_box_classes: np.ndarray of shape (?,) containing class num for
    box_predictions ordered by class and score

    predicted_box_scores: np.ndarray of shape (?) containing scores for
    box_predictions ordered by class and score
    """
        # Init empty lists to store final selected boxes, classes, and scores
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Sort boxes by class and then by score (in descending order)
        sorted_idxs = np.lexsort(((-box_scores), box_classes))

        # Iterate through sorted indices and apply NMS
        while len(sorted_idxs) > 0:

        #Get the current box's class and score using sorted indices
            curr_idx = sorted_idxs[0]
            current_box = filtered_boxes[curr_idx]
            current_class = box_classes[curr_idx]
            current_score = box_scores[curr_idx]

        # Add the current box, class, and score to the selected lists
            box_predictions.append(current_box)
            predicted_box_classes.append(current_class)
            predicted_box_scores.append(current_score)

        # Remove the current box from further consideration
            remaining_boxes = filtered_boxes[sorted_idxs[1:]]
        # Calculate IoU between the current box and remaining boxes
            iou = self.calc_iou(current_box, remaining_boxes)
        # Find indices of boxes with IoU <= threshold
            mask = np.where(iou <= self.nms_t)
            sorted_idxs = sorted_idxs[1:][mask]

        # Update boxes, scores, and classes based on IoU

        # Re-sort the remaining boxes by score (descending)

        # Return the final selected boxes, classes, and scores
        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def calc_iou(self, current_box, remaining_boxes):
        """calculates iou
        args:
            current_box: current bounding box w/ shape (4,)
                    x_min y_min x_max y_max

            remaining_boxes: boxes still under consideration before nms
                    shape(?, 4)"""
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        curr_box_area = (current_box[0] - current_box[2]) \
            * (current_box[1] - current_box[3])
        remaining_box_area = (remaining_boxes[:, 0] - remaining_boxes[:, 2]) \
            * (remaining_boxes[:, 1] - remaining_boxes[:, 3])

        union_area = curr_box_area + remaining_box_area - intersection

        IoU = intersection / union_area

        return IoU

    def preprocess_images(self, images):
        """ 
args:
    images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]

Returns a tuple of (pimages, image_shapes):
   
    pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3) 
    containing all of the preprocessed images
        ni: the number of images that were preprocessed
        input_h: input height for Darknet model Note: this can vary by model
        input_w: input width for Darknet model Note: this can vary by model
        3: number of color channels 

    image_shapes: a numpy.ndarray of shape (ni, 2)
    containing the original height and width of the images
        2 => (image_height, image_width)"""

        pimages = []
        image_shapes = []
        for image in images:
            original_h, original_w, _ = image.shape
            image_shapes.append((original_h, original_w))
            input = self.model.input
            input_h, input_w = input.shape[1:3]
            resized_image = cv2.resize(image, (input_h, input_w), interpolation=cv2.INTER_CUBIC)
            normalized_image = resized_image / 255.0
            pimages.append(normalized_image)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
args:
    image: a numpy.ndarray containing an unprocessed image

    boxes: a numpy.ndarray containing the boundary boxes for the image

    box_classes: a numpy.ndarray containing the class indices for each box

    box_scores: a numpy.ndarray containing the box scores for each box

    file_name: the file path where the original image is stored

    Displays the image with all boundary boxes, class names, and box scores
    Boxes should be drawn as with a blue line of thickness 2
    Class names and box scores should be drawn above each box in red
    Box scores should be rounded to 2 decimal places
    Text should be written 5 pixels above the top left corner of the box
    Text should be written in FONT_HERSHEY_SIMPLEX
    Font scale should be 0.5
    Line thickness should be 1
    You should use LINE_AA as the line type
    The window name should be the same as file_name
    If the s key is pressed:
    The image should be saved in the directory detections, located in the current directory
    If detections does not exist, create it
    The saved image should have the file name file_name
    The image window should be closed
    If any key besides s is pressed, the image window should be closed without saving"""

        for box, class_idx, score in zip(boxes, box_classes, box_scores):
            box = box.astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            class_name = self.class_names[class_idx]
            rounded_score = round(score)

            text_pos = (x1, y1 - 5)
            text = f"{class_name} {rounded_score}"
            cv2.putText(image, text, text_pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA)
            
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(f'detections/{file_name}', image)
            print(f"image saved as 'detections/{file_name}'")
        else:
            print("image not saved. window is closing")
        cv2.destroyAllWindows
    
    
    @staticmethod
    def load_images(folder_path):
        """
    args:
    folder_path: str representing path to folder holding the images to load

    Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images """

        #initializing lists
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            img = cv2.imread(file_path)

            images.append(img)
            image_paths.append(file_path)

        return images, image_paths
