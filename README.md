# Neural Networks starting with pure Python

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
