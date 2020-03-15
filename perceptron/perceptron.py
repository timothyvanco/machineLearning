import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    # X - training data, y - target output class labels - what network should predict
    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature matrix
        # this little trick allows to treat the bias as a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop for desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X, y):
                # dot product between input features and the weight matrix -> pass through the step func to obtain prediction
                p = self.step(np.dot(x, self.W))

                # perform weight update if prediction does not match the target
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x


    def predict(self, X, addBias=True):
        # ensure input is a matrix
        X = np.atleast_2d(X)

        # check if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        # dot product between input features and weight matrix
        return self.step(np.dot(X, self.W))


