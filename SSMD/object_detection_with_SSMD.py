import cv2 as cv
import numpy as np

def loadModel():
    # Loading MobileNetSSD
    network = cv.dnn.readNetFromCaffe("dnn/MobileNetSSD_deploy.prototxt",
                                       "dnn/MobileNetSSD_deploy.caffemodel")
    return network


def trainNetwork(image):
    blob = cv.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5 )
    #print(blob)

    net.setInput(blob)

    boxes = net.forward()
    #print(boxes)
    return boxes

def detection(image, boxes, detection_count, labels):


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

def labelName():
    # label name
    labels = []
    with open('ssmd_label.txt') as f:
        for line in f:
            labels.append(line.rstrip())
    return labels


if __name__ == "__main__":
    # Load network
    net = loadModel()

    cap = cv.VideoCapture(0)
   # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
   # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)
   # cap.set(cv.CAP_PROP_FPS, 100)

    while cap.isOpened():

        ret, frame = cap.read()

        boxes = trainNetwork(frame)

        detection_count = boxes.shape[2]

        labels = labelName()

        if ret == True:
            frame = detection(frame, boxes, detection_count, labels)
            cv.imshow('object detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()