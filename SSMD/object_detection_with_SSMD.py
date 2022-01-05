import cv2 as cv
import numpy as np

import sys
sys.path.append('E:/Object_Detection_with_Opencv_DNN/Config')
import Opencv_DNN

class OpenCvSSMD():

    """Object initialization"""
    def __init__(self,network,labelFile):
        self.network = network
        self.labelFile = labelFile

    def feedNetwork(self,image):
        blob = cv.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5 )
        #print(blob)
        #image = cv2.resize(image, (300, 300))
        #image = np.expand_dims(image, axis=0)
        #image = np.rollaxis(image, 3, 1)
        network.setInput(blob)
        boxes = network.forward()
        return boxes

    def objectDetection(self, image, boxes, detection_count, labels):

        COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

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
                #roi = frame[y: y2, x: x2]
                #roi_height = roi.shape[0]
                #roi_width = roi.shape[1]


                #"%s:%.2f" % (class_id, score)
                #lines = labelName()
                #cv.putText(image, lines[int(class_id)], (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #cv.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 3)

                print("class_id: {}, name: {}, score:{}".format(class_id, labels[int(class_id)], score))

                label = "{}: {:.2f}%".format(labels[int(class_id)], score * 100)
                cv.rectangle(image, (x, y), (x2, y2), COLORS[int(class_id)], 2)
                n_y = y - 15 if y - 15 > 15 else y + 15
                cv.putText(image, label, (x, n_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(class_id)], 2)

            else:
                continue

        return image

    """ get the class label name """

    def read_label(self):
        # label name
        lines = []
        with open(self.labelFile) as f:
            for line in f:
                lines.append(line.rstrip())
        return lines


if __name__ == "__main__":

    # Load network
    config_file = "dnn/MobileNetSSD_deploy.caffemodel"
    model_name = "dnn/MobileNetSSD_deploy.prototxt"
    labels = 'ssmd_label.txt'

    model = Opencv_DNN.ModelConfiguration(config_file, model_name)

    network = model.modelConfig("SSMD")

    if network is not None:
        OpencvSsmd = OpenCvSSMD(network, labels)

        cap = cv.VideoCapture(0)
       # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
       # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)
       # cap.set(cv.CAP_PROP_FPS, 100)

        while cap.isOpened():

            ret, frame = cap.read()

            boxes = OpencvSsmd.feedNetwork(frame)

            detection_count = boxes.shape[2]

            labels = OpencvSsmd.read_label()

            if ret == True:
                frame = OpencvSsmd.objectDetection(frame, boxes, detection_count, labels)
                cv.imshow('Detection with SSMD', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Please check network Configuration")