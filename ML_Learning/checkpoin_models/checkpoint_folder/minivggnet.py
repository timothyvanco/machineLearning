from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize model along with input shape to be "channeôs last"
        model = Sequential()
        inputShape = (height, width, depth)
        channelDimension = -1                           # -1 = last ordering

        # if "channels first" update input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            channelDimension = 1                        # batch normalization operatesover channels - in order to apply
                                                        # BN need to know which axis to normalize over, 1 = first order


        # first layer - (CONV => RELU => BN) * 2 => POOL => DO
        model.add(Conv2D(32, (3, 3), padding="same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))

        model.add(Conv2D(32, (3, 3), padding="same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))    # node from POOL layer will be randomly disconnected from next layer with prob 25%

        # second layer - (CONV => RELU => BN) * 2 => POOL => DO
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))                   # 512 nodes
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))                 # increasing probability to 50%

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    






