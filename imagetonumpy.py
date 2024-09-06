from imutils import paths
import os
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

class imagetonum():
    def build(imagePaths):
        data = []
        labels = []
        IMAGE_DIMS = (96, 96, 3)

        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            image = img_to_array(image)
            data.append(image)

        	# extract the class label from the image path and update the
        	# labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        print("[INFO] data matrix: {:.2f}MB".format(
        	data.nbytes / (1024 * 1000.0)))


        # binarize the labels
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        return data,labels

