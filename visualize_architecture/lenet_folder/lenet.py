# INPUT => CONV => TANH => POOL => CONV => TANH => POOL => FC => TANH => FC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # if using "channels first", update input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape)) # 20 filters, each of size 5x5
        model.add(Activation("relu"))                                        # ReLU activation function
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))            # 2x2 pooling with 2x2 stride - decreasing input volume size by 75%

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())        # input volume is flattened
        model.add(Dense(500))       # fully-connected layer with 500 nodes
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return constructed network architecture
        return model

    





