import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Loading Mask RCNN
net = cv.dnn.readNetFromTensorflow("custom_dnn/saved_model.pb",
                                    "custom_dnn/label_map.pbtxt")

img = cv.imread("custom_dnn/example.jpg")

height,width, channel = img.shape
print(img.shape)
print(height)


blob = cv.dnn.blobFromImage(img, swapRB = True)
#print(blob)
#print(blob.shape)
net.setInput(blob)

boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]

for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue

    # Get box Coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = img[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    cv.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

cv.imshow('object detection', img)
cv.waitKey(0)
cv.destroyAllWindows()