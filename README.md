# Neural Networks for MNIST digits starting with pure Python

![image of sample MNIST digits](images/digits.png)

The expected order to inspect/develop/test these examples is as given below.

The idea is to first develop some NN code that works in pure Python, and then continue to develop from there until we have a fairly comprehensive framework in pure Python supporting multiple layer types and a simple class-based declarative definition of the network.

On completion of the `mnist_conv` pure Python step of this development process we can define networks using our own classes such as:
```
net_conv = [
    Conv2D(3,1,8,{ "pad": 2, "stride": 2}),
    Pooling({"f": 3, "stride": 3, "mode": "max"}),
    ReLU(),
    Flatten(),
    Dense(200,10)
]
```

We can then discard our home-grown Python classes and move over to the tensorflow/keras API, now with a better comprehension of similarly defined networks such as:
```
model_lewis = Sequential()
model_lewis.add(Dense(256, input_shape=(784,), activation="relu"))
model_lewis.add(Dense(128, activation="relu"))
model_lewis.add(Dense(10, activation="softmax"))
```

## MNIST Weisberg - simplest neural network from scratch in Python

Build a single fully-connected-layer network with numpy to recognize hand-written
digits in the MNIST data set.

Ref: [Building a Neural Network from Scratch: Part 1](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/)

See source file in this repo `mnist_weisberg.py`

## MNIST (Aayush) Agrawal - restructure NN code using Python classes for each layer type

Similar to Weisberg, but abstracting the layers into Python classes.

There are copied and changed versions of this article but the reference below is preferred.

Ref: [Building Neural Network from scratch](https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9)

See source file in this repo `mnist_agrawal.py`

## MNIST Conv - add classes for Convolution

Adding new layer classes for
* Convolution
* Pooling (max & average)
* Softmax

## MNIST Keras

MNIST digit recognition using keras/tensorflow API

## Initial setup steps
### Build a Linux server

username james
hostname: james-dell5090
IP address 192.168.1.37
Netmask 255.255.255.0

adduser ijl20 / sudo

install openssh-server

check python

install python3-pip

sudo apt install python<version>-venv

make git repo / push to github

copy .gitignore from this repo

make/copy requirements.txt - include "wheel"

Make virtual env in development directory `python3 -m venv venv`

python3 -m pip install pip --upgrade

python3 -m pip install -r requirements.txt

### Jupyter use/config notes

run with
$ jupyter notebook

Set new password with

$ jupyter notebook password

Check config (e..g password) with

$ atom ~/.jupyter/jupyter_notebook_config.json

numpy

matplotlib
load image

display bar chart

display graph
