import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def trainNetwork(image):
    blob = cv.dnn.blobFromImage(image, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks

def detection(image, boxes, detection_count):
    height, width, channel = image.shape

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        if score > 0.5:
            # Get box Coordinates
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            # Region of Interest
            # roi = frame[y: y2, x: x2]
            # roi_height = roi.shape[0]
            # roi_width = roi.shape[1]

            "%s:%.2f" % (class_id, score)
            lines = labelName()
            cv.putText(image, lines[int(class_id)], (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            cv.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 3)
        else:
            continue
    return image

def labelName():
    # label name
    lines = []
    with open('label.txt') as f:
        for line in f:
            lines.append(line.rstrip())
    return lines

if __name__ == "__main__":
    # Loading Mask RCNN
    net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                        "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        boxes, masks = trainNetwork(frame)

        detection_count = boxes.shape[2]

        if ret == True:
            frame = detection(frame, boxes, detection_count)
            cv.imshow('object detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()