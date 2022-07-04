# Adding CONVOLUTIONAL layer to our class-based Python Neural Network
# Background see: https://www.youtube.com/watch?v=bNb2fEVKeEo
#
# See also the notebook overview linked here, from which the conv layer functions are cut-and-paste:
# https://github.com/enggen/Deep-Learning-Coursera/blob/master/Convolutional%20Neural%20Networks/Week1/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb

# This builds upon the python class-based dense NN code derived from:
# https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

# Typical usage:
# >>> exec(open("mnist_agrawal2.py").read())
# >>> train(net_agrawal)
# >>> predict(net_agrawal, [ X_test[0] ])

# Note that training does NOT use the X_test images in any way

# Check 1st 10 test predictions:
# for i in range(10):
#     p = predict(net_agrawal,[X_test[i]])
#     print(f"Image {i} predict {p} vs {y_test[i]}")

from __future__ import print_function
import numpy as np ## For numerical python
import tensorflow.keras as keras
from tqdm import trange
from IPython.display import clear_output

import matplotlib.pyplot as plt
# %matplotlib inline                 # UNCOMMENT for JUPYTER

np.random.seed(42)

# ********************************************************************************************
# ******** Here is a 'template' for the layer classes ****************************************
# ********************************************************************************************

class Layer:

    #A building block. Each layer is capable of performing two things:
    #- Process input to get output:           output = layer.forward(input)

    #- Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    #Some layers also have learnable parameters which they update during layer.backward.

    def __init__(self):
        # Here we can initialize layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass

    def forward(self, input):
        # Takes input data of shape [batch, input_units], returns output data [batch, output_units]

        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        # Performs a backpropagation step through the layer, with respect to the given input.
        # To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        # d loss / d x  = (d loss / d layer) * (d layer / d x)
        # Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        # If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input) # chain rule

class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass

    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward

    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output * relu_grad # This relies on the quirk Python True === 1, False === 0

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b

        self.learning_rate = learning_rate

        # initialize weights with small random numbers. We use normal initialization
        self.weights = np.random.normal(loc=0.0,
                                        scale = np.sqrt(2/(input_units+output_units)),
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)

    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b

        # input shape: [batch, input_units]
        # output shape: [batch, output units]

        return np.dot(input,self.weights) + self.biases

    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

# *********************************************************************
# ***** Convolutional layer                                       *****
# ***** Origin: https://github.com/enggen/Deep-Learning-Coursera  *****
# *********************************************************************

# E.g. X_test.shape is (10000,784)
# X_test[0].shape is (784,)
# X_test_conv = X_test.reshape(-1,28,28.1)
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, img_height, img_width, img_depth) representing:
        m : image batch size
        img_height x img_width : pixel h x w = size of images
        img_depth : colors e.g. RGB = 3
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    ### END CODE HERE ###

    return X_pad

# GRADED FUNCTION: conv_single_step

def conv_single_step(img_patch, filter_W, filter_b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    Note filter size is f x f, with the same depth as the input image
    img_patch -- slice of input data of shape (f, f, depth_in)
    filter_W -- Weight parameters contained in a window - matrix of shape (f, f, depth_in)
    filter_b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    #print("conv_single_step img_patch:", img_patch.shape)
    #print(img_patch)

    #print("conv_single_step filter_W:", filter_W.shape)
    #print(filter_W)

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(img_patch, filter_W)

    #print("conv_single_step s:", s.shape)
    #print(s)

    # Sum over all entries of the volume s.
    Z = np.sum(s)
    #print("conv_single_step Z:", Z)

    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(filter_b)
    ### END CODE HERE ###

    return Z

# Note here we're use for loops rather than Python vectorization
def conv_forward(input, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    input -- output activations of the previous layer,
             numpy array of shape (count, height_in, width_in, depth_in)
    W -- Weights, numpy array of shape (f, f, depth_in, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (count, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    ### START CODE HERE ###
    # Retrieve dimensions from input's shape (≈1 line)
    (count, height_in, width_in, depth_in) = input.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, depth_in, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((height_in - f + 2 * pad) / stride) + 1
    n_W = int((width_in - f + 2 * pad) / stride) + 1

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((count, n_H, n_W, n_C))
    # Define matrix A, output after activation here A = np.array(Z.shape)

    # Create input_pad by padding input
    input_pad = zero_pad(input, pad)

    for i in range(count):                                 # loop over the batch of training examples
        example_img = input_pad[i]                     # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the (3D) slice of example_img (See Hint above the cell). (≈1 line)
                    img_patch = example_img[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(img_patch, W[...,c], b[...,c])
                    # Add activation here: A[i, h, w, c] = activation(Z[i, h, w, c])
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(Z.shape == (count, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (input, W, b, hparameters)

    # Alternatively return A
    return Z, cache

# Note call with e.g.
# np.random.seed(1)
# A_prev = np.random.randn(10,4,4,3)
# E.g. for 2 x 2 x 3 filters, x8:
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2,
#               "stride": 2}

# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
def conv():
    W = np.random.randn(3,3,1,8)
    b = np.random.randn(1,1,1,8)
    hparams = { "pad": 2, "stride": 2}
    X_train_conv = X_train.reshape(-1,28,28,1)
    Z, cache = conv_forward(X_train_conv, W, b, hparams)

# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3,2,1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

# Max or Average POOLING layer

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    ### END CODE HERE ###

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

# Call with e.g. A, cache = pool_forward(A_prev, hparameters, mode = "max")

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):                       # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

# Call with e.g. dA, dW, db = conv_backward(Z, cache_conv)

# *********************************************************************
# ******* Cross-entropy forwards and backwards         ****************
# *********************************************************************

def softmax_crossentropy_with_logits(logits,y):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    logits_for_answers = logits[np.arange(len(logits)),y]
    return - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

def grad_softmax_crossentropy_with_logits(logits,y):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    ones_for_answers = np.zeros_like(logits) # create zeroes matrix same shape as logits
    ones_for_answers[np.arange(len(logits)),y] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]

# *********************************************************************
# ********* MNIST DATA LOAD               *****************************
# *********************************************************************

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test

# Compute activations of all network layers by applying them sequentially.
# Return a LIST of LISTS: activations for each layer.
def forward(network, X):

    activations = []
    input = X
    # Looping through each layer passing output of each as input of next
    for layer in network:
        activations.append(layer.forward(input))
        # Updating input to last layer output
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

# predict(network layer list, samples array)
# X.shape is (N,784), i.e. a list of flattened MNIST images
# logits.shape is (N,10)
# return shape is (N,) i.e. a list of categories (0..9), one for each input image
def predict(network,X):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = forward(network,X)[-1] # forward() returns the entire network stack of output values, so the final [-1] set is net output

    return logits.argmax(axis=-1) # argmax returns the index of the highest element

# infer() is the same as predict(), but for a single image and printing more readable results
def infer(network, image):
    logits = forward(network,[image])[-1] # as predict(), converting image into a list
    # logits.shape is (1,10), i.e. the output of the final layer for a single image
    print_image(image)
    print_logits(logits[0])
    print_category(logits.argmax(axis=-1))

def train_batch(network,X,y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.


    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  # NOTE "+" here concatenates [X] and layer_activations. layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)

    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]

        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates

    return np.mean(loss)

# An untested function to work out if we're running in IPython/Jupyter or not (returns False ok in Python command line)
def in_ipython():
    try:
        x = get_ipython()
        return True
    except NameError:
        return False

# Python ITERATOR that yields batchsize slices of X and y
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ************************************************************
# ******* Network TRAIN function                  ************
# ************************************************************
def train(network, batchsize=32, shuffle=True):
    # We'll accumulate a log of accuracy figures for the training dataset and validation data set
    train_accuracy = []
    val_accuracy = []

    for epoch in range(8): # Was 25 epochs
        print("Epoch",epoch)

        # With default batch size of 32, and 50,000 samples, this will iterate through 1562 batches
        # iterate_minibatches simply chops up X_train, y_train into 32-
        for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=batchsize,shuffle=shuffle):
            train_batch(network,x_batch,y_batch)

        train_accuracy.append(np.mean(predict(network,X_train)==y_train))
        val_accuracy.append(np.mean(predict(network,X_val)==y_val))

        if in_ipython():
            clear_output()
        print("Train accuracy:",train_accuracy[-1])
        print("Val accuracy:",val_accuracy[-1])
        if in_ipython():
            plt.plot(train_accuracy,label='train accuracy')
            plt.plot(val_accuracy,label='val accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

    ## Let's look at some example
    if in_ipython():
        plt.figure(figsize=[6,6])
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.title("Label: %i"%y_train[i])
            plt.imshow(X_train[i].reshape([28,28]),cmap='gray');

# ***************************************************
# ********* Utility functions                ********
# ***************************************************

# Print a 784 x 1 image to the console
def print_image(image):
    for i in range(28):
        for j in range(28):
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

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

print(f"X_train is {X_train.shape}")
print(f"X_val is {X_val.shape}")
print(f"X_test is {X_test.shape}")
print("X_test[0] is:")
print_image(X_test[0])

net_agrawal = [
    Dense(784,100),
    ReLU(),
    Dense(100,200),
    ReLU(),
    Dense(200,10)
]

net_lewis = [
    Dense(784,100),
    ReLU(),
    Dense(100,10)
]

# train(net_agrawal)
