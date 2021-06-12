import cv2
import numpy as np
import time
import argparse

def _load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") #cv2.dnn.readNet(): chay chuyen tiep / yolov3.weight : YOLO :https://pjreddie.com/darknet/yolo/
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()] # trả về các chỉ số của 
                                                                                #các lớp đầu ra của mạng 
    colors = np.random.uniform(0,255, size = (len(classes), 3))
    return net, classes, colors, output_layers

def start_webcam():
    return cv2.VideoCapture(0)