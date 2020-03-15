from lenet_folder import LeNet
from tensorflow.keras.utils import plot_model

# visualize LeNet and then write the network architecture visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)
