import cv2 as cv
import numpy as np

import sys
sys.path.append('E:/Object_Detection_with_Opencv_DNN/Config')
import Opencv_DNN

class OpenCVYOLOV4():

    """Object initialization"""
    def __init__(self, network, labelFile):
        self.network = network
        self.labelFile = labelFile

    def feedNetwork(self, image, net):
        blob = cv.dnn.blobFromImage(image, 1/255, (416, 416),  swapRB=True)
        #print(blob)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        #print(layerNames)
        #print(layerNames[199])
        unConnectedOutputLayers = net.getUnconnectedOutLayers()
        #print(unConnectedOutputLayers)

        outputNames = [layerNames[i-1] for i in unConnectedOutputLayers]
        #print(outputNames)

        outputs = net.forward(outputNames)
        #print(len(outputs[0]), len(outputs[1]), len(outputs[2]))
        return outputs

    def objectDetection(self, image, outputs, labels):

        COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

        height, width, channel = image.shape

        b_boxes = []
        box_confidences = []
        class_ids = []

        for output in outputs:
            #print('shape: {}'.format(output.shape))
            for box in output:
                #print(box)
                scores = box[5:]
                #print(scores)
                #print("scores len: {}".format(len(scores)))
                class_id = np.argmax(scores)
                #print(class_id)
                score = scores[class_id]
                #print("Score: {}".format(confidence))

                if score > 0.6:
                    w = int(box[2] * width)
                    h = int(box[3] * height)
                    x = int((box[0] * width) - w/2)
                    y = int((box[1] * height) - h/2)

                    b_boxes.append([x,y,w,h])
                    class_ids.append(class_id)
                    box_confidences.append(float(score))

        #print(len(b_boxes))
        #scoreThreshold = 0.6, nmsThreshold = 0.4
        indices = cv.dnn.NMSBoxes(b_boxes, box_confidences, 0.6, 0.4)

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
    config_file = "dnn/yolov4.cfg"
    model_name = "dnn/yolov4.weights"
    label_file = 'yolo_label.txt'

    model = Opencv_DNN.ModelConfiguration(config_file, model_name)

    network = model.modelConfig("YOLOV4")

    if network is not None:
        OpencvYoloV4 = OpenCVYOLOV4(network, label_file)

        cap = cv.VideoCapture(0)
       # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
       # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)
       # cap.set(cv.CAP_PROP_FPS, 100)

        while cap.isOpened():

            ret, frame = cap.read()

            outputs = OpencvYoloV4.feedNetwork(frame, network)

            labels = OpencvYoloV4.readLabel()

            if ret == True:
                frame = OpencvYoloV4.objectDetection(frame, outputs, labels)
                cv.imshow('object detection', frame)
                #break
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Please check network Configuration")