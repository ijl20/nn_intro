import tensorflow.keras as keras
import numpy as np

def load_dataset(flatten=False):
    """
    Returns:
    images_train - training data     60000x784
    labels_train - training labels   60000x1
    images_test  - test data         10000x784
    labels_test  - test labels       10000x1

    The source data loaded via keras holds the images as 28x28 - the 'flatten' arg
    will convert this to 784x1.

    images_train is initially 60000x28x28, and images_test is 10000x28x28

    The images are 'flattened' to 784x1, and also the pixels are normalized from
    0..255 to 0..1
    """
    # Use keras library to load MNIST data from interweb
    (images_train, labels_train), (images_test, labels_test) = keras.datasets.mnist.load_data()
    print("Loaded training images", images_train.shape)
    print("Loaded test images", images_test.shape)

    # normalize input pixel values int 0..255 to float 0..1
    images_train = images_train.astype(float) / 255.
    images_test = images_test.astype(float) / 255.

    if flatten:
        images_train = images_train.reshape([images_train.shape[0], -1])
        images_test = images_test.reshape([images_test.shape[0], -1])

    return (images_train, labels_train), (images_test, labels_test)


def zero_one(y_train):
    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == 0.0)[0]] = 1
    return y_new

# Print a 784 x 1 image to the console
def print_image(image):
    for i in range(28):
        for j in range(28):
            print( '  ' if image[i*28+j] > 0.5 else '00', end='')
        print()

# SIGMOID activation function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# LOSS function

def compute_loss(Y, Y_hat):

    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )

    return L

def run():
    learning_rate = 1

    (X_train, y_train),(X_test, y_test) = load_dataset()

    X = X_train.reshape([X_train.shape[0],-1]).T # 60000 x 28 x 28 -> 60000 x 784

    print("Input matrix X is",X.shape)

    n_x = X.shape[0]
    print("n_x is", n_x)
    m = X.shape[1]
    print("m is ", m)

    Y = zero_one(y_train).reshape(1,m)

    print("Truth matrix Y is", Y.shape)

    # Create 784x1 weight matrix initialised with small random numbers
    W = np.random.randn(n_x, 1) * 0.01
    print("Weight matrix W is ", W.shape)

    # Bias is 1 x 1
    b = np.zeros((1, 1))

    for i in range(2000):
        Z = np.matmul(W.T, X) + b
        A = sigmoid(Z)

        cost = compute_loss(Y, A)

        dW = (1/m) * np.matmul(X, (A-Y).T)
        db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)

        W = W - learning_rate * dW
        b = b - learning_rate * db

        if (i % 100 == 0):
            print("Epoch", i, "cost: ", cost)

    print("Final cost:", cost)
