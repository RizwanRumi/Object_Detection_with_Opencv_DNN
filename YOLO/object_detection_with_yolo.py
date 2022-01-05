import cv2 as cv
import numpy as np

def loadModel():
    # Loading YOLOV3
    network = cv.dnn.readNetFromDarknet("dnn/yolov3.cfg", "dnn/yolov3.weights")
    network.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return network


def feedForward(image, net):
    #print("------Feed Forward------")
    blob = cv.dnn.blobFromImage(image, 1/255, (320, 320), [0,0,0], crop=False)
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
    #print("----------------")
    return outputs

def detectObject(image, outputs, labels):

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

            if score > 0.5:
                w = int(box[2] * width)
                h = int(box[3] * height)
                x = int((box[0] * width) - w/2)
                y = int((box[1] * height) - h/2)

                b_boxes.append([x,y,w,h])
                class_ids.append(class_id)
                box_confidences.append(float(score))

    #print(len(b_boxes))
    #scoreThreshold = 0.5, nmsThreshold = 0.3
    indices = cv.dnn.NMSBoxes(b_boxes, box_confidences, 0.5, 0.3)

    for i in indices:
        box = b_boxes[i]
        x,y,w,h = box[0], box[1], box[2], box[3]

        #cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.rectangle(image, (x, y), (x + w, y + h), COLORS[int(class_id)], 2)
        '''
        cv.putText(image, f'{labels[class_ids[i]].upper()} {int(box_confidences[i] * 100)}%',
                    (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        '''
        label = "{}: {:.4f}".format(labels[class_ids[i]], box_confidences[i])
        n_y = y - 15 if y - 15 > 15 else y + 15
        cv.putText(image, label, (x, n_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(class_id)], 1)

    return image


def labelName():
    # label name
    labels = []
    with open('yolo_label.txt') as f:
        for line in f:
            labels.append(line.rstrip())
    return labels


if __name__ == "__main__":
    # Load network
    net = loadModel()
    #print(net)

    cap = cv.VideoCapture(0)
   # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
   # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)
   # cap.set(cv.CAP_PROP_FPS, 100)

    while cap.isOpened():

        ret, frame = cap.read()

        outputs = feedForward(frame, net)

        #detection_count = boxes.shape[2]

        labels = labelName()

        if ret == True:
            frame = detectObject(frame, outputs, labels)
            cv.imshow('object detection', frame)
            #break
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()