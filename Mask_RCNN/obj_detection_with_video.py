import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('E:/Object_Detection_with_Opencv_DNN/Config')
import Opencv_DNN

class OpencvMaskRcnn:

    """ OpenCV Mask-Rcnn takes four parameter for the initialization object"""

    def __init__(self, net, labelFile):
        self.net = net
        self.labelFile = labelFile

    """ feed the input towards the network """

    def feed_network(self, image):
        # detect object
        blob = cv.dnn.blobFromImage(image)
        self.net.setInput(blob)
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        return boxes, masks

    """ get the class label name """

    def read_label(self):
        # label name
        lables = []
        with open(self.labelFile) as f:
            for line in f:
                lables.append(line.rstrip())
        return lables

    """ bounding box detection and ROI"""

    def objectDetection(self, image, boxes, detectionCount):
        img_height = image.shape[0]
        img_width = image.shape[1]

        labelNames = self.read_label()

        for i in range(detectionCount):
            box = boxes[0, 0, i]
            class_id = box[1]
            # print(class_id)
            score = box[2]
            if score < 0.5:
                continue
            # Get the box Coordinates
            x = int(box[3] * img_width)
            y = int(box[4] * img_height)
            x2 = int(box[5] * img_width)
            y2 = int(box[6] * img_height)
            # find the Region of interest
            roi = image[y: y2, x: x2]

            label = "{} : {:.4f}".format(labelNames[int(class_id)], score)
            cv.putText(image, label, (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            roi_height, roi_width, _ = roi.shape
            cv.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 3)

        return image


if __name__ == "__main__":

    config_file = "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    model_name = "dnn/frozen_inference_graph_coco.pb"
    labels = 'label.txt'

    model = Opencv_DNN.ModelConfiguration(config_file, model_name)

    net = model.modelConfig("MASKRCNN")

    OpencvMaskRCNN = OpencvMaskRcnn(net, labels)


    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        boxes, masks = OpencvMaskRCNN.feed_network(frame)
        detection_count = boxes.shape[2]

        if ret == True:
            frame = OpencvMaskRCNN.objectDetection(frame, boxes, detection_count)
            cv.imshow('Detection with Mask-R-CNN', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()