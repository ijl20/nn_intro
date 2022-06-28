# Neural network intro

## Build a Linux server
username james
hostname: james-dell5090
IP address 192.168.1.37
Netmask 255.255.255.0

adduser ijl20 / sudo

install openssh-server

check python

install python3-pip

sudo apt install python<version>-venv

copy .gitignore

make git repo / push to github

make requirements.txt - include "wheel"

python3 -m pip install pip --upgrade

## Jupyter

pip install jupyter (add to requirements.txt)

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

## MNIST Weisberg

Build a single fully-connected-layer network with numpy to recognize hand-written
digits in the MNIST data set.

Ref: [Building a Neural Network from Scratch: Part 1](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/)

See source file in this repo `mnist_weisberg.py`

## MNIST (Aayush) Agrawal

Similar to Weisberg, but abstracting the layers into Python classes.

There are copied and changed versions of this article but the reference below is preferred.

Ref: [Building Neural Network from scratch](https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9)

See source file in this repo `mnist_agrawal2.py`
