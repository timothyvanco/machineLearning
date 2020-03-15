from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		self.dataFormat = dataFormat		# store the image data format

	def preprocess(self, image):
		return img_to_array(image, data_format=self.dataFormat) # Keras func that correctly rearranges dimensions of image

	
