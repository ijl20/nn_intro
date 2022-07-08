# Adding CONVOLUTIONAL layer to our class-based Python Neural Network
# Background see: https://www.youtube.com/watch?v=bNb2fEVKeEo
#
# See also the notebook overview linked here, from which the conv layer functions are cut-and-paste:
# BAD EXAMPLE ? https://github.com/enggen/Deep-Learning-Coursera/blob/master/Convolutional%20Neural%20Networks/Week1/Convolution%20model%20-%20Step%20by%20Step%20-%20v2.ipynb
# https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/blob/master/Convolutional%20Neural%20Networks/week1/convolution_model.py

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

# *********************************************************************************
# ***** Convolutional layer                                                   *****
# ***** https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera  *****
# *********************************************************************************

class Conv2D(Layer):
    def __init__(self, filter_size, input_depth, filter_count, hparams, learning_rate=0.1):

        self.learning_rate = learning_rate

        # Initialize conv weights 3x3 (x1 color) (x8 filters)
        self.weights = np.random.randn(filter_size,filter_size,input_depth,filter_count)

        # Each filter has one bias weight per input_depth
        self.biases = np.random.randn(1,1,input_depth,filter_count)

        self.hparams = hparams

        print(f"Conv2D initialized with {filter_count} filters, each {filter_size}x{filter_size}x{input_depth}")
        print(f'Conv2D params are pad={hparams["pad"]}, stride={hparams["stride"]}')

    def forward_single_step(self, input_example_slice, filter_W, filter_b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
        of the previous layer.

        Arguments:
        Note filter size is f x f, with the same depth as the input image
        input_example_slice -- slice of input data of shape (f, f, depth_in)
        filter_W -- Weight parameters contained in a window - matrix of shape (f, f, depth_in)
        filter_b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, self.biases) on a slice x of the input data
        """

        #print("conv_single_step input_example_slice:", input_example_slice.shape)
        #print(input_example_slice)

        #print("conv_single_step filter_W:", filter_W.shape)
        #print(filter_W)

        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between input_example_slice and filter_W. Do not add the bias yet.
        s = np.multiply(input_example_slice, filter_W)

        #print("conv_single_step s:", s.shape)
        #print(s)

        # Sum over all entries of the volume s.
        Z = np.sum(s)
        #print("conv_single_step Z:", Z)

        # Add bias filter_b to Z. Cast filter_b to a float() so that Z results in a scalar value.
        Z = Z + float(filter_b)
        ### END CODE HERE ###

        return Z

    def forward(self,input):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        input -- output activations of the previous layer,
                 numpy array of shape (count, height_in, width_in, depth_in)
        self.weights -- Weights, numpy array of shape (f, f, depth_in, filter_count)
        self.biases -- Biases, numpy array of shape (1, 1, 1, filter_count)
        self.hparams -- python dictionary containing "stride" and "pad"

        Returns:
        Z -- conv output, numpy array of shape (count, output_height, output_width, filter_count)
        cache -- cache of values needed for the conv_backward() function
        """

        #print("Conv2D forward called with input",input.shape)
        ### START CODE HERE ###
        # Retrieve dimensions from input's shape (≈1 line)
        (count, height_in, width_in, depth_in) = input.shape

        # Retrieve dimensions from self.weights's shape (≈1 line)
        (f, f, depth_in, filter_count) = self.weights.shape

        # Retrieve information from "hparameters" (≈2 lines)
        stride = self.hparams['stride']
        pad = self.hparams['pad']

        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        output_height = int((height_in - f + 2 * pad) / stride) + 1
        output_width = int((width_in - f + 2 * pad) / stride) + 1

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((count, output_height, output_width, filter_count))
        # Define matrix A, output after activation here A = np.array(Z.shape)

        # Create input_pad by padding input
        input_pad = self.zero_pad(input, pad)

        for i in range(count):                                 # loop over the batch of training examples
            input_example = input_pad[i]                     # Select ith training example's padded activation
            for h in range(output_height):                           # loop over vertical axis of the output volume
                for w in range(output_width):                       # loop over horizontal axis of the output volume
                    for c in range(filter_count):                   # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the (3D) slice of input_example (See Hint above the cell). (≈1 line)
                        input_example_slice = input_example[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter weights and biases, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = self.forward_single_step(input_example_slice, self.weights[...,c], self.biases[...,c])
                        # Add activation here: A[i, h, w, c] = activation(Z[i, h, w, c])
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert(Z.shape == (count, output_height, output_width, filter_count))

        # Save information in "cache" for the backprop
        # cache = (input, self.weights, self.biases, hparameters)

        # Alternatively return A
        return Z

    def backward(self, input, grad_output):
        """
        Implement the backward propagation for a convolution function

        Arguments:

        grad_output -- gradient of the cost with respect to the output of the conv layer (Z),
                       shape (count, output_height, output_width, filter_count)

        Returns:
        grad_input -- gradient of the cost with respect to the input of the conv layer (input),
                   numpy array of shape (count, input_height, input_width, input_depth)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, input_depth, filter_count)
        db -- gradient of the cost with respect to the biases of the conv layer (self.biases)
              numpy array of shape (1, 1, 1, filter_count)
        """

        ### START CODE HERE ###

        # Retrieve dimensions from input's shape
        (count, input_height, input_width, input_depth) = input.shape

        # Retrieve dimensions from W's shape
        (f, f, input_depth, filter_count) = self.weights.shape

        # Retrieve information from "hparameters"
        stride = self.hparams["stride"]
        pad = self.hparams["pad"]

        # Retrieve dimensions from grad_output's shape
        (count, output_height, output_width, filter_count) = grad_output.shape # Note count was already loaded above, should be same number

        # Initialize grad_input, dW, db with the correct shapes
        grad_input = np.zeros((count, input_height, input_width, input_depth))
        dW = np.zeros((f, f, input_depth, filter_count))
        # db = np.zeros((1, 1, 1, filter_count)) # ?? Assumes filter depth == 1 ??
        db = np.zeros((1, 1, input_depth, filter_count)) # If input depth > 1 ??

        # Pad input and grad_input
        input_pad = self.zero_pad(input, pad)
        d_input_pad = self.zero_pad(grad_input, pad)

        for i in range(count):                       # loop over the training examples

            # select ith training example from input_pad and d_input_pad
            input_example_pad = input_pad[i]      # input_example_pad = input_pad[i, :, :, :]
            d_example_pad = d_input_pad[i]    # d_example_pad = d_input_pad[i, :, :, :]

            for h in range(output_height):                   # loop over vertical axis of the output volume
                for w in range(output_width):               # loop over horizontal axis of the output volume
                    for c in range(filter_count):           # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = stride * h
                        vert_end = vert_start + f
                        horiz_start = stride * w
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from input_example_pad
                        a_slice = input_example_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        d_example_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.weights[:,:,:,c] * grad_output[i, h, w, c]
                        dW[:,:,:,c] += a_slice * grad_output[i, h, w, c]
                        db[:,:,:,c] += grad_output[i, h, w, c]

            # Set the ith training example's grad_input to the unpaded d_example_pad (Hint: use X[pad:-pad, pad:-pad, :])
            grad_input[i, :, :, :] = d_example_pad[pad:-pad, pad:-pad, :]
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert(grad_input.shape == (count, input_height, input_width, input_depth))

        return grad_input, dW, db

# Call with e.g. grad_input, dW, db = conv_backward(Z, cache_conv)


    # E.g. X_test.shape is (10000,784)
    # X_test[0].shape is (784,)
    # X_test_conv = X_test.reshape(-1,28,28.1)
    def zero_pad(self, X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
        as illustrated in Figure 1.

        Argument:
        X -- python numpy array of shape (count, img_height, img_width, img_depth) representing:
            count : image batch size
            img_height x img_width : pixel h x w = size of images
            img_depth : colors e.g. RGB = 3
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions

        Returns:
        X_pad -- padded image of shape (count, n_H + 2*pad, n_W + 2*pad, n_C)
        """

        ### START CODE HERE ### (≈ 1 line)
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        ### END CODE HERE ###

        return X_pad


# GRADED FUNCTION: conv_single_step


# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3,2,1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


# *********************************************************************
# ***** POOLING Layer (Max or Average)                            *****
# *********************************************************************

class Pooling(Layer):
    def __init__(self, hparams):
        """
        hparams -- python dictionary containing "f", "stride" and "mode":
            f: the pooling window will be f x f
            stride: amout window will be stepped vertically and horizontally
            mode -- the pooling mode "max" or "average"
        """
        self.mode = hparams["mode"]
        self.f = hparams["f"]
        self.stride = hparams["stride"]

        print(f"Pooling layer initialized with mode={self.mode}")
        print(f'Pooling layer hparams are f={self.f}, stride={self.stride}')

    def forward(self, input):
        """
        Implements the forward pass of the pooling layer

        Arguments:
        input -- Input data, numpy array of shape (count, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the pool layer, a numpy array of shape (count, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input
        """

        # Retrieve dimensions from the input shape
        (count, n_H_prev, n_W_prev, n_C_prev) = input.shape


        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((count, n_H, n_W, n_C))

        ### START CODE HERE ###
        for i in range(count):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        # Use the corners to define the current slice on the ith training example of input, channel c. (≈1 line)
                        input_slice = input[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(input_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(input_slice)

        ### END CODE HERE ###

        # Store the input and hparameters in "cache" for pool_backward()
        # cache = (input, hparameters)

        # Making sure your output shape is correct
        assert(A.shape == (count, n_H, n_W, n_C))

        return A

        # Call with e.g. A, cache = pool_forward(input, hparameters, mode = "max")

    def create_mask_from_window(self, x):
        """
        Used for MAX Pooling.
        Creates a mask from an input matrix x, to identify the max entry of x.
        Arguments:
        x -- Array of shape (f, f)
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """

        ### START CODE HERE ### (≈1 line)
        mask = (x == np.max(x))
        ### END CODE HERE ###

        return mask

    def distribute_value(self, dz, shape):
        """
        Used for AVERAGE Pooling.
        Distributes the input value in the matrix of dimension shape
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """

        ### START CODE HERE ###
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape

        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)

        # Create a matrix where every entry is the "average" value (≈1 line)
        a = average * np.ones(shape)
        ### END CODE HERE ###

        return a

    def backward(self, input, grad_output):
        """
        Implements the backward pass of the pooling layer
        Arguments:
            input -- output from the forward pass of the pooling layer, contains the layer's input
            grad_output -- gradient of cost with respect to the output of the pooling layer, same shape as input

        Returns:
            d_input -- gradient of cost with respect to the input of the pooling layer, same shape as input
        """

        #print(f"Pooling.backward called input {input.shape}, grad_output {grad_output.shape}")

        # Retrieve information from cache (≈1 line)
        #(input, hparameters) = cache

        # Retrieve dimensions from input's shape and grad_output's shape (≈2 lines)
        count, n_H_prev, n_W_prev, n_C_prev = input.shape
        count, n_H, n_W, n_C = grad_output.shape

        # Initialize d_input with zeros (≈1 line)
        # d_input = np.zeros((count, n_H, n_W, n_C)) # is this a BUG? Should be input shape
        d_input = np.zeros((count, n_H_prev, n_W_prev, n_C_prev))

        for i in range(count):  # loop over the training examples

            # select training example from input (≈1 line)
            example = input[i, :, :, :]

            for h in range(n_H):  # loop on the vertical axis
                for w in range(n_W):  # loop on the horizontal axis
                    for c in range(n_C):  # loop over the channels (depth)

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = self.stride * h
                        vert_end = vert_start + self.f
                        horiz_start = self.stride * w
                        horiz_end = horiz_start + self.f

                        # Compute the backward propagation in both modes.
                        if self.mode == "max":

                            # Use the corners and "c" to define the current slice from example (≈1 line)
                            example_slice = example[horiz_start:horiz_end, vert_start:vert_end, c]
                            #print(f"Pooling.backward input[{i}] {input[i].shape}, example_slice {example_slice.shape}")
                            # Create the mask from example_slice (≈1 line)
                            mask = self.create_mask_from_window(example_slice)
                            #print(f"Pooling.backward mask {mask.shape}")
                            #print(f"Pooling.backward Updating d_input {d_input.shape}")
                            # Set d_input to be d_input + (the mask multiplied by the correct entry of grad_output) (≈1 line)
                            d_input[i, vert_start:vert_end, horiz_start:horiz_end, c] += example_slice * mask

                        elif self.mode == "average":

                            # Get the value a from grad_output (≈1 line)
                            da = np.sum(example[horiz_start:horiz_end, vert_start:vert_end, c])
                            # Define the shape of the filter as fxf (≈1 line)
                            filter_shape = (self.f, self.f)
                            # Distribute it to get the correct slice of d_input. i.e. Add the distributed value of da. (≈1 line)
                            d_input[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, filter_shape)

        ### END CODE ###

        # Making sure your output shape is correct
        assert (d_input.shape == input.shape)

        return d_input

# *********************************************************************
# ***** FLATTEN                                                   *****
# *********************************************************************

class Flatten(Layer):
    def __init__(self):
        """
        Provides forwards/backwards flatten a (count,A,B,C,..) multi-dimensional input into (count,N)
        Particularly suited to flattening the output of convolutional layers
        """
        print(f"Flatten layer initialized")

    def forward(self, input):
        """
        Implements the forward pass of the pooling layer
        """
        self.shape = input.shape
        return input.reshape(input.shape[0],-1)

    def backward(self, input, grad_output):
        """
        Implements backward pass of flatten layer
        """

        # assumes a forward pass has preceded this backward call
        return grad_output.reshape(self.shape)

# *********************************************************************
# ***** SOFTMAX                                                   *****
# *********************************************************************

# https://e2eml.school/softmax.html
# https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function

class Softmax(Layer):
    def __init__(self):
        """
        Forwards/backwards compute of Softmax function
        """
        print(f"Softmax layer initialized")

    def forward(self,input):
        """Compute the softmax of count vectors."""
        count, width = input.shape
        output = np.empty((count,width))
        for i in range(count):
            exps = np.exp(input[i] - input[i].max())
            sm = exps / np.sum(exps)
            output[i,:] = sm
        return output

    def backward(self, input, grad_output):
        """
            input - input provided on most recent forward pass
            grad_output - gradients presented upward from next layer
        """
        print(f"Softmax.backward called input {input.shape}, grad_output {grad_output.shape}")
        count, width = input.shape
        # Init the array we will return
        input_grad = np.empty((count,width))
        for i in range(count):
            # recalculate softmax output (could cache this), reshape to single row
            exps = np.exp(input[i] - input[i].max())
            sm = exps / np.sum(exps)
            grads = grad_output[i].reshape(1,-1) # also reshape to single row
            d_softmax = (sm * np.identity(sm.size) - sm.transpose() @ sm)
            example_grads = grads @ d_softmax
            #print(f"Softmax.backward example_grads {example_grads.shape}")
            input_grad[i,:] = example_grads
        return input_grad

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

    #print("train_batch")

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
def train(network, X_train, y_train, X_val, y_val, batchsize=32, shuffle=True):
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

# Print any 2D H x W matrix to the console
def print_2D(filter_output):
    (h,w) = filter_output.shape
    for y in range(h):
        for x in range(w):
            if (filter_output[y,x]<0):
                print(f"{filter_output[y,x]:.3f} ",end="")
            else:
                print( f"{filter_output[y,x]:.3f} ",end="")
        print()

# Print list of logits
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
#print("X_test[0] is:")
#print_image(X_test[0])

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

net_conv = [
    Conv2D(3,1,8,{ "pad": 2, "stride": 2}),
    Pooling({"f": 3, "stride": 3, "mode": "max"}),
    ReLU(),
    Flatten(),
    Dense(200,10)
]

def conv_forward(X_train_conv):

    # Create 3x3x1 conv filters x8
    layer1 = Conv2D(3,1,8,{ "pad": 2, "stride": 2})
    #Z, cache = layer.forward(X_train_conv)
    Z1 = layer1.forward(X_train_conv)
    print("Conv2D forward pass complete, output shape", Z1.shape) # (100,15,15,8)
    print("Output of filter 0 for image 0 is:")
    print_conv(Z1[0,:,:,0])

    # Create a max-pooling layer
    layer2 = Pooling({"f": 3, "stride": 3, "mode": "max"})
    Z2 = layer2.forward(Z1)
    print("Pooling forward pass complete, output shape", Z2.shape) # (100,5,5,8)
    print("Max Pooled filter 0 for image 0 is:")
    print_conv(Z2[0,:,:,0])

    # create ReLU layer
    layer3 = ReLU()
    Z3 = layer3.forward(Z2)
    print("ReLU forward pass complete, output shape", Z3.shape) # (100,5,5,8)
    print("ReLU filter 0 for image 0 is:")
    print_conv(Z3[0,:,:,0])

    # Create Flatten layer
    layer4 = Flatten()
    Z4 = layer4.forward(Z3)
    print("Flatten forward pass complete, output shape", Z4.shape) # (100,200)

    # Create Dense layer
    layer5 = Dense(200,10)
    Z5 = layer5.forward(Z4)
    print("Dense forward pass complete, output shape", Z5.shape) # (100,10)

    # Create Softmax layer
    layer6 = Softmax()
    Z6 = layer6.forward(Z5)
    print("Softmax forward pass complete, output shape", Z6.shape) # (100,10)

    return Z6

# Test train on first 100 images
X_train_conv = X_train.reshape(-1,28,28,1)
y_train_conv = y_train

X_test_conv = X_test.reshape(-1,28,28,1)
y_test_conv = y_test

X_val_conv = X_val.reshape(-1,28,28,1)[:200]
y_val_conv = y_val[:200]


# train(net_agrawal, X_train, y_train, X_val, y_val)
# train(net_conv, X_train_conv, y_train_conv, X_val_conv, y_val_conv)
