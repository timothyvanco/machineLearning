import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of of weight matrices
        # store network architecture and learning rate
        self.W = []
        self.layers = layers    # list of integers - represent architecture of NN
        self.alpha = alpha      # learning rate

        # start looping from the index of the first layer
        # stop before reach last 2 layers
        for i in np.arange(0, len(layers) - 2): # from zero to stop before last 2
            # randomly initialize a weight matrix connecting the number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1) # '1' in both are for bias
            self.W.append(w / np.sqrt(layers[i]))

        # last two layers are a special case where the input connections needs a bias
        # but last dont
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represent network architecture
        return "NeuralNetwork: {}".format("-".join(str(layer) for layer in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    # fit function responsible for training NN
    # X - training data, y - labels
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1's as the last entry in the feature matrix -> bias as trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over every individual datapoint in training set, make prediction, compute backpropagation
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check if to display training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))


    # x - individual data point from matrix
    # y - corresponding class label
    def fit_partial(self, x, y):
        # construct list of output activations for each layer as data point flows through the network;
        # the first activation is a special case -- it's just the input feature vector itself
        A = [np.atleast_2d(x)]      # View inputs as arrays with at least two dimensions

        # FEEDFORWARD
        # loop over the layers in network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking the dot product between the activation and
            # the weight matrix -- "net input" to the current layer
            net = A[layer].dot(self.W[layer])

            # compute "net output"
            out = self.sigmoid(net)
            A.append(out)


        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference between *prediction* (the final output
        # activation in the activations list) and the true target value
        error = A[-1] - y       # array[-1] - want to acces last entry in the list

        # here need to apply the chain rule and build list of deltas `D`; the first entry in the deltas is
        # simply the error of the output layer times the derivative of activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])] # deltas are used to update weight matrices scaled by learning rate

        # loop over the layers in reverse order
        for layer in np.arange(len(A) - 2, 0, -1):
            # delta for the current layer == to the delta of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta, by the derivative of the nonlinear
            # activation function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since looped over layers in reverse order need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            # update weights by taking dot product of the layer activations with respective deltas
            # then multiplying this value by some small learning rate and adding to wieght matrix
            # this is where "learning" take place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer]) # <- gradient descent

    def predict(self, X, addBias=True):
        # initialize output prediction as input features
        # this value will be (forward) propagated through the network to obtain final prediction
        predictValue = np.atleast_2d(X)

        # check if bias column should be added
        if addBias:
            predictValue = np.c_[predictValue, np.ones((predictValue.shape[0]))]         # bias trick

        for layer in np.arange(0, len(self.W)):
            predictValue = self.sigmoid(np.dot(predictValue, self.W[layer]))

        return predictValue

    def calculate_loss(self, X, targets):
        # make predictions for input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss




