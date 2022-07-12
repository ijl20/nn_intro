# Machine Learning Glossary

Training data
* The data (e.g. images) used to train a neural network.

Validation data
* The data (e.g. images) kept separate from the training data, used after training to assess the accuracy of the model using images
not seen before.

Test data
* Sometimes used to mean 'Validation data' as above,
* sometimes used to refer to a users 'own' data (e.g. images) that
are used to test the neural network after development using the training data and validation data is completed.

Weights
* The parameters within the neural network model that are trained (i.e. adjusted) as new data annotated with the correct
'ground truth' is fed into the model. Typically these weights are used as simple multipliers of the input data, i.e. much
of the math in a neural network is simple linear multiplication. The complex behaviour of neural networks results from the
use of a very large number of weights, i.e. millions would be common.

Batch
* When you first start with NN's, you tend to think of it in terms of the network 'iterating' through the input
examples during the training phase. This works fine for your comprehension of how the nework 'works', but in practice
the actual implementation of all neural networks assumes the input data being provided a 'batch' at a time (i.e. there
is one more dimension to the input data for the rows of input examples). This allows the matrix math to crunch through
the training data incredibly fast, taking advantage of execution optimizations spanning multiple input examples.
The `batch_size` is consequently yet another NN meta-parameter you might tune to improve the effectiveness of your NN.

Epoch
* This is typically a single run through all of the input examples, at which point summary stats for the whole data
set can be collected or reported (such as training accuracy). Training will typically be for a number of epochs, with the
general idea the accuracy will improve a bit on each run through the sample data until no further gains are achievable at
which point the NN generally starts getting worse. So knowing when to stop training is important but the optimal 'epochs count'
meta-parameter is commonly found by trial and error.

Image classification
* The process of using a neural network to partition input images into a pre-defined set of image classes (e.g.
'CAT', 'DOG'...).

Overfitting
* The behaviour of a neural network as its prediction accuracy for the training data begins to far exceed the
accuracy it is achieving on the separate validation data

Label
* The 'class' or 'category' given in the training and validation data for each input value. If the training and test
values are referred to as 'x' values, the label is often referred to as the corresponding 'y' value.

One-hot encoding
* If a value needs to encode which class out of N classes to which an input value belongs, this could simply
be represented by an integer 0..N-1 representing the assigned class. However, for some uses (in particular training
a neural network) this class membership is more conveniently represented as an array of N numbers in which all the
values are 0 except for a 1 in the position representing the chosen class (e.g. [0,0,1,0,0] would represent class
membership in the third class of 5 possible classes). This 'one-hot' representation is good for comparison with a
similar output from a neural network where the class membership might be uncertain, e.g. [0.1,0.2,0.5,0.1,0.1].

Inference Pass, also Predict Pass
* This is a 'simple' forward pass through the neural network, with a data value (or set of data values) provided as
input. For a successfully trained neural network, this will result in the output from the neural network providing a
good classification of the input.

Prediction
* The output of the neural nework after being given a sample input value. E.g. given a picture of a cat, the
neural network may produce a vector of 10 output values, each representing the probability the picture is of a member of
each of 10 pre-defined classes. In practice, neural networks, particularly during training, are given whole batches of
input data and produce a corresponding 'batch' or predictions.

Activation function
* The default behaviour of a neural network layer will be to output some derived matrix resulting from its input.
Due to compounded additions or multiplications, the range of values in this output may diverge greatly from
the range of values in the input (e.g. the input may have every value in the range 0..1) which might cause overflow
or inaccuracies if simply propagated to later layers in the network. An activation function (for example ReLU or
sigmoid) can be applied to the output of the layer to 'normalise' the data in some way to avoid the possible
runaway effect of the values increasing (or decreasing) substantially as the data propagates through the network.

ReLU
* An activation function that can be applied to the output of a neural network layer which replaces negative output
values with zero and leaves positive output values unchanged.

Sigmoid
* An activation function that smoothly squashes layer output values into the range -1..+1.

Softmax
* An activation function that normalizes output values to between 0..1 with a sum of 1.0, such that they can be
interpreted as probabilities of class/category membership. Typically used on the final output layer of a neural
network.

Optimizer
* The algorithm used to find the minimum loss while iterating through the training data, e.g. Stochastic Gradient Descent
or Adam.

(Stochastic) Gradient Descent
* The effect of the backpropagation algorithm that gradually shifts the weights in the neural network in a
direction that improves its overall accuracy. I.e. the network is slowly descending down some error gradient until
it reaches its minimum error. The use of 'stochastic' refers to the random choice of training data values. This is a
type of optimizer.

Loss function
* A calculation comparing the output of a neural network during training with the given 'correct' answer. A greater
'loss' value implies the output is more 'wrong'. For a scalar output value, the loss calculation can be simple such
as the squared differnce of the output value vs. the given correct value.  A more complex function will be used when
the training labels are one-hot encoded and the neural network has a softmax-encoded output layer.

Cross-entropy
* A type of loss function useful for comparing one-hot encoded training labels with a softmax output layer of a
neural network.

Logit
* An obscure word typically used to mean the output values of a neural network layer, often the final layer that can
be visualized as a bar chart representing the predicted class membership probabilities.

Shape
* After the Python `my_matrix.shape` method, which you will use thousands of times during your NN development to confirm that
the output of one layer of your network is what you intended, e.g. the output `A` of a convolutional layer may give
`A.shape = (None,24,24,8)` representing the output of a conv layer with 8 24x24 filters, given an arbitrarily long batch input data.

Dense
* A type of neural network layer that is fully connected to the layer above, i.e. every cell in the dense layer will
have a value that is the product of weight multiples of every output from the layer above.

Deep
* Multi-layer

Multi-layer
* see Deep

Meta-parameters
* The various arbitrary numbers used to define a neural network such as the number of layers, the width of dense layers,
the choice of activation functions, and the choice and order of layer types.

Neural Network Cookery
* The application of labor in adjusting meta-parameters while searching for an improved accuracy of the trained network.

Image classification
* The use of a trained neural network to assign a predefined class identifier to previously unseen images, e.g. 'CAT'.

Epoch
* During training of the neural network, a complete run through the input training data set is referred as an 'epoch'.
Although the training process will run through the input data many time (sometimes randomizing the order of data selection)
this breakpoint is often convenient to pause and accumulate accuracy statistics than can subsequently be graphed so that an
observer can see how the accuracy of the network is improving during the training runs. Most often it gives a clue to
the developer to kill the process as the code is clearly heading in the wrong direction.

Keras
* A popular Python API in which neural networks can be defined. Those networks can be run on multiple neural network
'engines' that are designed to efficiently execute the matrix math embedded in the neural network code, such as
Tensorflow.

Convolution
* A general matrix operation commonly applied to images in which a smaller matrix (called a filter or kernel)
can be considered iterated over a larger matrix, producing an output matrix in which each element is a summed
linear multiplication of the matching elements of the two input matrixes. See also Zero padding.

Stride
* See Convolution. The filter (e.g. 7x7) being moved across the larger input image (e.g. 224x224) does not have to
move across in single-pixel steps. Maybe that would provide more detail than necessary particularly if the filter
is expected to do something simple like pick up on diagonal lines. In this case the filter can be hopped across the
image in larger steps, referred to as the stride, reducing the data and computation required.

Convolution kernel
* Also called a convolution filter. A matrix, typically smaller than a chosen input matrix, which is used to convolve
that original image (see Convolution). A non-obvious result is that even a small kernel can provide a significant
enhancement of information contained within the original input image, e.g. a kernel can highlight features in the
image such as human eyes or cats' whiskers. This simplifies the task of subsequent layers in a neural network in
categorizing images. A remarkable insight is that the weights embedded in the kernel can themselves be learned
during the training process of a neural network (the matrix math of the convolution process is very similar to
that of a fully-connected layer). This means a neural network with convolutional layers is in practice learning
for itself which important features to pick out in the provided training images.

Zero padding
* See Convolution. A slight complexity occurs at the edges of the larger input matrix as there are multiple options for
what to do as the convolution filter bumps up against the edge of the input image. in the simplest as the input image
can be convolved as-is, and the output image will necessarily be slightly smaller than the input image. The convolution
function typically includes options to pad the input matrix with 1's or 0's around the edge, or other options.

MNIST
* An ancient digital source of monochrome 28x28 pixel handwritten digits useful for training your first neural network.

sklearn also scikit-learn
* An awesome open-source library of Python data science functions (https://scikit-learn.org/stable/)

matplotlib
* A Python library to draw charts and images, only slightly more complicated than the neural network you are designing.

numpy
* A Python numerical library that makes Python the best teaching language for anything involving matrixes, especially
neural networks.

cv2 also opencv
* An open source image processing library. When you re-enter the real world after experimenting with artificial example
neural networks, you will inevitably need to maniulate your images using opencv before they are fit to be passed to
the fragile world of neural networks. In fact it is possible your image recognition requirement (such as face detection)
is more effectively solved using traditional methods supported by opencv than using neural networks.
In general, however, opencv is likely to provide a useful image-processing pipeline that is used to front your
neural network to get the image into some normalized state for better analysis (such as normalized brightness
and contrast, or discarding bad video frames) (https://opencv.org).

Batch size
* Neural networks are trained on chunks of input data at a time, commonly referred to as 'batches'. E.g. with images,
perhaps 100 (or any number) my be passed to the neural nework in each batch, the neural network will produce a
similarly-sized batch of 'predictions'

Flatten
* A neural network layer that changes the shape of the data from the layer above, typically to change two dimensions (as
in an image) into a single dimension (in which all the pixels are strung out into a line). This is a necessary
pre-step if the data is then to be passed to a Dense layer.

Max pooling
* A simple way to reduce the quantity of data output by a layer, before passing it to the next layer. If the prior
layer has two-dimensional values (e.g. as can be considered input images or convolved images) then a typical operation
is to take 2x2 blocks of values from the prior layer output and replace each block with a single value (the max of those four)
in the new output. That operation will replace the size of those two dimensions by a factor of 2.  A max pooling operation
(or layer) will have meta-parameters defining the size of the block to be reduced and also the 'step' size as the pooling
operator steps across the input data.

Tensorflow
* Google's neural network execution engine.  Version 1.x included Google's own usage API while Version 2.0 onwards
(still beta in 2019) recommends the Keras API as the strategic choice. Tensorflow, like other engines, is designed
to be fast and can partially execute the neural network algorithms using GPU's.

Fully-connected Layer
* See Dense Layer

Locally-connected Layer
* A less-common network layer function which is a derivative of the convolution layer and the dense layer (which themselves
are very similar when considered in terms of the matrix math involved).  The simple description of the locally-connected
layer is that it is similar to the convolutional layer except each 'patch' of the source image is considered to have
its own filter, rather than a 'shared' filter that can be thought of as scrolling across the input image.

Backpropagation
* The training process by which the 'loss' value is propagated backwards through the neural nework and each layer's
weights are adjusted by very small amounts in a direction that will reduce the overall error. This process depends
upon the fact that the forward functions though the network (such as the linear multipliers in the Dense or Convolution
layers, or the activation functions) are differentiable. It is this fact that allows the direction of adjustment of each
weight can be calculated and the incremental training steps can move the entire network towards the global minimum.

Edge detection
* This refers to the use of a convolution kernel that results in the highlighting of edges in an input image.

Sobel operator
* An example of a convolution kernel which effectively highlights edges in an input image.

Object recognition
* A superset of Image Classification, in which objects are recognized anywhere they appear in an input image and both
the object classifications and locations (typically as a box around each recognized object) are provided as outputs. A
typical use-case is a transport camera in which pedestrians, bicycles, cars and trucks can be separately counted.

Adversarial image
* Neural nets trained on a large number of example images will learn some formula which separates those images into the
desired categories such as 'cat' or 'dog', but the internal representation defined within the millions of trained weights will not
reflect the typical properties a human would use to 'define' a cat or a dog. It is possible for some image looking absolutely nothing
like a cat to be confidently predicted to be 'cat' due to the input reinforcing the formula through the learned weights in some
unexpected way. That image may be referred to as 'adversarial' given they could be manufactured specifically to fool a given
trained network.

Accuracy
* A general term hinting at how often did the NN get the answer 'right', but a simple answer rarely tells the full story.
See 'Top-5 Accuracy' and 'Confusion Matrix'.

Top-5 Accuracy
* If the ground truth for a given input example is some class, and a NN predicts probability-weighted classes for that input example,
and the correct answer is anywhere in the NN's top 5, that scores as a win under the Top-5 rules. This has the unfortunate effect
that a NN categorizing a beach-ball as (1) Battleship, (2) Antelope, (3) Orange, (4) Moon, (5) Beach-ball ... scores a +1. Hopefully
this comparative metric has been consigned to the trash-can of history.

Confusion matrix
* This is a fancy term for a simple presentation of accuracy, i.e. for each category in the classifier show what percentage of
the true objects in that category were reported as such (correct positives) and what percentage were wrongly reported (false
positives). This allows reasonable comprehension of accuracy (i.e. a NN correctly identified a racoon 98% the time) but doesn't
include a concept of how spectacularly wrong were the predictions the remaining 2% of the time.

Overfitting
* This is the idea that if you train a very large network over and over again on the same data, it will learn quirks of that data
(for example it might take a short-cut detecting faces of specific glasses-wearers) which works against the intention for
generalization. The symptom of overfitting is a reduction in the accuracy with new test data.

Dropout
* One method to reduce the likelihood of overfitting, discarding random weights during the training process to add an element of
chaos.
