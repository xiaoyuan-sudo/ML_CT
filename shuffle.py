from imutils import paths
import os
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array


class dataconstruct:
	def generate(path):
# initialize the data and labels
		data = []
		labels = []
		IMAGE_DIMS = (96, 96, 3)

		# grab the image paths and randomly shuffle them
		print("[INFO] loading images...")
		imagePaths = sorted(list(paths.list_images(path)))
		random.seed(42)
		random.shuffle(imagePaths)
		k = int(np.round(len(imagePaths)*0.9))

		traindata = imagePaths[0:k]
		testdata = imagePaths[k:]
		return traindata, testdata
