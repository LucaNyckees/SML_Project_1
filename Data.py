import tensorflow as tf
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def GenerateData(k,l,number):

    Input_Data = []
    Output_Data =[]
    i = 0
    j = 0
    while i < number:
        if (y_train[j] == k or y_train[j]==l) :
            local=list(itertools.chain(*x_train[j]))
            Input_Data.append(local)
            Output_Data.append(y_train[j])
            i=i+1
        j=j+1
    Data=[Input_Data, Output_Data]

    return Data


def GenerateLabeledData(type, number):

    Data = []
    i = 0
    j = 1
    while i < number:
        if (y_train[j] == type) :
            Data.append(x_train[j])
            i=i+1
        j=j+1

    return Data


def GenerateUnlabeledData(number):

    Data = []
    i = 0
    j = 1
    while i < number:
        if (y_train[j] == 1 or y_train[j]==2) :
            Data.append(x_train[j])
            i=i+1
        j=j+1

    return Data


def DisplayData(data):

   fig = plt.figure(figsize=(8, 8))
   columns = round(math.sqrt(len(data))) + 1
   rows = round(math.sqrt(len(data))) + 1
   for i in range(1, len(data)+1):
       img = data[i - 1]
       fig.add_subplot(rows, columns, i)
       plt.xticks([])
       plt.yticks([])
       plt.imshow(img, cmap=plt.cm.binary)

   plt.show()


def TransformData(data):

    Transformed_Data = []
    for i in range(0, len(data)):
            local = list(itertools.chain(*data[i]))
            Transformed_Data.append(local)

    return Transformed_Data


def ResizeData(size, data):

    newdata = []
    for i in range(0,len(data)):
        resized_img = cv2.resize(data[i], (size, size))
        newdata.append(resized_img)

    return newdata
