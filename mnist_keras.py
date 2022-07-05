
# https://pyimagesearch.com/2021/05/06/implementing-feedforward-neural-networks-with-keras-and-tensorflow/

# *****************************
# ** Instructions for use:
# *****************************

# Load into python with:
# exec(open("mnist_keras.py").read())
# then can generate loss/accuracy curves with:
# train(model_rosebrock)
# history = evaluate(model_rosebrock)
# plot(history, "mnist_keras.png")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
#	help="path to the output loss/accuracy plot")
# args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# each image in the MNIST dataset is represented as a 28x28x1
# image, but in order to apply a standard neural network we must
# first "flatten" the image to be simple list of 28x28=784 pixels
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 784-256-128-10 architecture using Keras
model_rosebrock = Sequential()
model_rosebrock.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model_rosebrock.add(Dense(128, activation="sigmoid"))
model_rosebrock.add(Dense(10, activation="softmax"))

# Another model
model_lewis = Sequential()
model_lewis.add(Dense(256, input_shape=(784,), activation="relu"))
model_lewis.add(Dense(128, activation="relu"))
model_lewis.add(Dense(10, activation="softmax"))

def train(model, epochs=100, batch_size=128):
    # train the model using SGD
    print("[INFO] training network...")
    sgd = SGD(0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=sgd,
    	          metrics=["accuracy"])
    H = model.fit(trainX, trainY,
                     validation_data=(testX, testY),
    	             epochs=epochs,
                     batch_size=batch_size)

    return H.history

def evaluate(model, batch_size=128):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=batch_size)
    print(classification_report(testY.argmax(axis=1),
    	predictions.argmax(axis=1),
    	target_names=[str(x) for x in lb.classes_]))

def plot(history, filename):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    x_count = len(history["loss"])
    plt.plot(np.arange(0, x_count), history["loss"], label="train_loss")
    plt.plot(np.arange(0, x_count), history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, x_count), history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, x_count), history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(filename)

# ***************************************************
# ********* Utility functions                ********
# ***************************************************

# Print a 784 x 1 image to the console
def print_image(image, h=28, w=28):
    for i in range(h):
        for j in range(w):
            print( '  ' if image[i*28+j] > 0.5 else '00', end='')
        print()

# Print logits
def print_logits(logits):
    for n in logits:
        if (n<0):
            print(f"{n:.3f} ",end='')
        else:
            print(f" {n:.3f} ",end='')
    print()

# Print category, e.g. 0 -> "ZERO"
# Python 3.10 only
def print_category(n):
    match n:
        case 0:
            print("ZERO")
        case 1:
            print("ONE")
        case 2:
            print("TWO")
        case 3:
            print("THREE")
        case 4:
            print("FOUR")
        case 5:
            print("FIVE")
        case 6:
            print("SIX")
        case 7:
            print("SEVEN")
        case 8:
            print("EIGHT")
        case 9:
            print("NINE")
        case _:
            print("ERROR")

# infer() is the same as predict(), but for a single image and printing more readable results
def infer(model, image):
    logits = model.predict(np.array([image]))[0] # as predict(), converting image into a list
    # logits.shape is (1,10), i.e. the output of the final layer for a single image
    print_image(image)
    print_logits(logits)
    print_category(logits.argmax())
