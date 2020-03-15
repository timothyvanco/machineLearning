from backpropagation import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# load the MNIST dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1]
# (each image is represented by an 8 x 8 = 64-dim feature vector)

print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
#  min/max normalizing by scaling each digit into the range [0, 1]
data =  (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])   # architecture of NN - output layer 10 because of possibility 0-9 digits
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))



