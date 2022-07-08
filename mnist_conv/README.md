# MNIST Python Convolution, Pooling, Softmax

The idea here is to extend our previous `mnist_agrawal` Python Class-based framework to include support for Convolutional layers.

This extends the Coursera/deeplearning.ai Course #4 Module 1 example which is discussed
[here](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/).

Good (non-Class-based) code is available
[here](https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/tree/master/Convolutional%20Neural%20Networks/week1)
although possibly with minor bugs. This code does not include a complete network definition or a training framework.

`mnist_conv_class.py` defines classes for each of the new layers, and defines a new network `net_conv` using those:
```
net_conv = [
    Conv2D(3,1,8,{ "pad": 2, "stride": 2}),
    Pooling({"f": 3, "stride": 3, "mode": "max"}),
    ReLU(),
    Flatten(),
    Dense(200,10)
]
```

This network uses 28x28x1 images, where the earlier networks used (784,), so `mnist_conv_class.py` also reshapes the earlier X_train
etc. into
```
X_train_conv, y_train_conv
X_val_conv, y_val_conv
X_test_conv, y_test_conv
```

Source can be loaded into Python (and the MNIST data loaded) with:
```
exec(open("mnist_conv_class.py").read())
```

A test single 'forward' run can be tested with:
```
conv_forward(X_train_conv)
```

Full training of the network can be completed with (this will take a while):
```
train(net_conv, X_train_conv, y_train_conv, X_val_conv, y_val_conv)
```

A forward pass on a single image can be performed with:
```
predict(net_conv, np.array([X_train_conv[0]]))
```
Or similarly,  set of images in the data set can be tested with:
```
predict(net_conv, X_test_conv[0:10,:,:,:])
```
This may return something like:
```
array([3, 0, 4, 1, 9, 2, 1, 3, 1, 4])
```
while the truth values can be displayed:
```
y_train_conv[0:10]
```
giving
```
array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4], dtype=uint8)
```
In the above example, the network was correct except for the first example.
