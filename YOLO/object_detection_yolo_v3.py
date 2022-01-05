import cv2 as cv
import numpy as np

import sys
sys.path.append('E:/Object_Detection_with_Opencv_DNN/Config')
import Opencv_DNN

class OpenCVYOLOV3():

    """Object initialization"""
    def __init__(self, network, labelFile):
        self.network = network
        self.labelFile = labelFile

    def feedNetwork(self, image, net):
        blob = cv.dnn.blobFromImage(image, 1/255, (320, 320), [0,0,0], crop=False)
        #print(blob)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        unConnectedOutputLayers = net.getUnconnectedOutLayers()
        outputNames = [layerNames[i-1] for i in unConnectedOutputLayers]
        outputs = net.forward(outputNames)
        return outputs

    def objectDetection(self, image, outputs, labels):
        COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
        height, width, channel = image.shape

        b_boxes = []
        box_confidences = []
        class_ids = []

        for output in outputs:
            for box in output:
                scores = box[5:]
                class_id = np.argmax(scores)
                score = scores[class_id]

                if score > 0.5:
                    w = int(box[2] * width)
                    h = int(box[3] * height)
                    x = int((box[0] * width) - w/2)
                    y = int((box[1] * height) - h/2)

                    b_boxes.append([x,y,w,h])
                    class_ids.append(class_id)
                    box_confidences.append(float(score))

        indices = cv.dnn.NMSBoxes(b_boxes, box_confidences, 0.5, 0.3)

        for i in indices:
            box = b_boxes[i]
            x,y,w,h = box[0], box[1], box[2], box[3]

            cv.rectangle(image, (x, y), (x + w, y + h), COLORS[int(class_id)], 2)
            label = "{}: {:.4f}".format(labels[class_ids[i]], box_confidences[i])
            cv.putText(image, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(class_id)], 1)

        return image

    def readLabel(self):
        # label name
        lines = []
        with open(self.labelFile) as f:
            for line in f:
                lines.append(line.rstrip())
        return lines


if __name__ == "__main__":
    # Load network
    config_file = "dnn/yolov3.cfg"
    model_name = "dnn/yolov3.weights"
    label_file = 'yolo_label.txt'

    model = Opencv_DNN.ModelConfiguration(config_file, model_name)

    network = model.modelConfig("YOLOV3")

    if network is not None:
        OpencvYoloV3 = OpenCVYOLOV3(network, label_file)

        cap = cv.VideoCapture(0)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)
        # cap.set(cv.CAP_PROP_FPS, 100)

        while cap.isOpened():

            ret, frame = cap.read()

            outputs = OpencvYoloV3.feedNetwork(frame, network)

            labels = OpencvYoloV3.readLabel()

            if ret == True:
                frame = OpencvYoloV3.objectDetection(frame, outputs, labels)
                cv.imshow('object detection', frame)
                # break
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Please check network Configuration")