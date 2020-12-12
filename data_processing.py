import tensorflow as tf
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def ResizeData(size, data):

    newdata = []
    for i in range(0,len(data)):
        resized_img = cv2.resize(data[i], (size, size))
        newdata.append(resized_img)

    return newdata


def TransformData(data):

    Transformed_Data = []
    for i in range(0, len(data)):
            local = list(itertools.chain(*data[i]))
            Transformed_Data.append(local)

    return Transformed_Data


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

def GenerateDataforDisplay(k,l,number,size):

    Input_Data = []
    Output_Data =[]
    i = 0
    j = 0
    while i < number:
        if (y_train[j] == k or y_train[j]==l) :
            Input_Data.append(x_train[j])
            Output_Data.append(y_train[j])
            i=i+1
        j=j+1
    Input_Data=ResizeData(size,Input_Data)
    Data=[Input_Data, Output_Data]

    return Data

def GenerateData(k,l,number,size):

    Input_Data = []
    Output_Data =[]
    i = 0
    j = 0
    while i < number:
        if (y_train[j] == k or y_train[j]==l) :
            Input_Data.append(x_train[j])
            Output_Data.append(y_train[j])
            i=i+1
        j=j+1
    Input_Data=ResizeData(size,Input_Data)
    Input_Data=TransformData(Input_Data)
    Data=[Input_Data, Output_Data]

    return Data

""""
ok_test=GenerateData(1,2,20,8)
print(ok_test[0][0])

ok_test2=GenerateDataforDisplay(1,2,20,8)
print(ok_test2[0])
DisplayData(ok_test2[0])
"""

