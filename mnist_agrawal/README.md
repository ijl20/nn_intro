# MNIST Agrawal

This MNIST digit classification moves us on from `mnist_weisberg` which was the simplest in-line Python code, to a very similar implementation with the NN layers defined within Python Classes.

With these class definitions the network can be more abstractly declared as below:
```
net_agrawal = [
    Dense(784,100),
    ReLU(),
    Dense(100,200),
    ReLU(),
    Dense(200,10)
]
```

Other network designs can be tested, for example:
```
net_lewis = [
    Dense(784,100),
    ReLU(),
    Dense(100,10)
]
```

## Future improvements

The `train()` function should be passed the data arrays to train on, rather than being hardcoded to use the `X_train`, `y_train`, `X_val`, `y_val` globals.

The softmax/cross-entropy code is embedded as function calls in the train function, this might be better provided as a Layer class.

Having implemented the later Conv layers in Python (see mnist_conv), it would probably be better to consistently use the images
as count x 28x28x1 rather than count x (784,). This would mean bringing forward the (simple) `Flatten` layer from mnist_conv into
this example.
