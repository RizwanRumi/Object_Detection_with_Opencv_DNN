import cv2 as cv

class ModelConfiguration():

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def modelConfig(self, dnn_name):
        if dnn_name == "MASK_RCNN":
            network = cv.dnn.readNetFromTensorflow(self.model, self.config)
        elif dnn_name == "SSMD":
            network = cv.dnn.readNetFromCaffe(self.model, self.config)
        elif(dnn_name == "YOLOV4"):
            network = cv.dnn.readNetFromDarknet(self.config, self.model)
        else:
            network = None

        network.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        return network
