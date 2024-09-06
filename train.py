# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimage.pyimage import SmallerVGGNet
from alexnet import MyAlexNet
from keras.callbacks import Callback
import matplotlib.pyplot as plt 
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import pandas as pd
import keras
import tensorflow as tf

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 400
INIT_LR = 1e-4
BS =  16
IMAGE_DIMS = (96, 96, 3)
 
data = np.load('train_x.npy')
labels = np.load('train_y.npy')

class EarlyStoppingValAcc(Callback):
    def __init__(self, monitor='val_acc', value=0.80, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.value:

            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

callbacks = [EarlyStoppingValAcc(monitor='val_accuracy', value=0.85,verbose=1)]
# initialize the model
print("[INFO] compiling model...")
model = MyAlexNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"]) 



# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY,epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('pokedex.model')



# plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
N = EPOCHS
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="upper left")
#plt.savefig('plot.png')
np.savetxt('train_loss.txt',H.history["loss"])
#np.savetxt('val_loss.txt',H.history["val_loss"])
np.savetxt('train_acc.txt',H.history["accuracy"])
#np.savetxt('val_acc.txt',H.history["val_accuracy"])
model.summary()
