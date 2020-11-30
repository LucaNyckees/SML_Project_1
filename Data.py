
def GenerateLabeledData(type, number):
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    Data = []
    i = 0
    j = 1
    while i < number:
        if (y_train[j] == type) :
            Data.append(x_train[j])
            i=i+1
        j=j+1
    return Data

def DisplayData(data):
   import math
   import matplotlib.pyplot as plt
   fig = plt.figure(figsize=(8, 8))
   columns = round(math.sqrt(len(data))) + 1
   rows = round(math.sqrt(len(data))) + 1
   for i in range(1, len(data)):
       img = data[i - 1]
       fig.add_subplot(rows, columns, i)
       plt.imshow(img, cmap=plt.cm.binary)
   plt.show()


def GenerateUnlabeledData(number):
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    Data = []
    i = 0
    j = 1
    while i < number:
        if (y_train[j] == 1 or y_train[j]==2) :
            Data.append(x_train[j])
            i=i+1
        j=j+1
    return Data

ok1=GenerateUnlabeledData(120)
ok=GenerateLabeledData(2,100)
DisplayData(ok)
DisplayData(ok1)