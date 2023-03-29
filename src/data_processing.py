import tensorflow as tf
import math
import matplotlib.pyplot as plt
import cv2
import itertools

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def resize_data(size: int, data: list) -> list:
    newdata = []
    for i in range(0, len(data)):
        resized_img = cv2.resize(data[i], (size, size))
        newdata.append(resized_img)

    return newdata


def transform_data(data: list) -> list:
    Transformed_Data = []
    for i in range(0, len(data)):
        local = list(itertools.chain(*data[i]))
        Transformed_Data.append(local)

    return Transformed_Data


def display_data(data: list) -> None:
    fig = plt.figure(figsize=(8, 8))
    columns = round(math.sqrt(len(data))) + 1
    rows = round(math.sqrt(len(data))) + 1
    for i in range(1, len(data) + 1):
        img = data[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

    plt.show()


def generate_data_for_display(k, l, number, size) -> list:
    input_data = []
    output_data = []
    i = 0
    j = 0
    while i < number:
        if y_train[j] == k or y_train[j] == l:
            input_data.append(x_train[j])
            output_data.append(y_train[j])
            i = i + 1
        j = j + 1
    input_data = resize_data(size, input_data)
    Data = [input_data, output_data]

    return Data


def generate_data(k, l, number, size) -> list:
    input_data = []
    output_data = []
    i = 0
    j = 0
    while i < number:
        if y_train[j] == k or y_train[j] == l:
            input_data.append(x_train[j])
            output_data.append(y_train[j])
            i = i + 1
        j = j + 1
    input_data = resize_data(size, input_data)
    input_data = transform_data(input_data)
    data = [input_data, output_data]

    return data
