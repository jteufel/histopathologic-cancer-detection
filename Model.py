import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import pandas as pd
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

#load in data

trainLabelsCSV = pd.read_csv('./histopathologic-cancer-detection-data/train_labels.csv')
trainLabels = trainLabelsCSV['label']

trainFiles = glob('./histopathologic-cancer-detection-data/train/*.tif')
testFiles = glob('./histopathologic-cancer-detection-data/test/*.tif')

#Data preprocessing

trim = 5000 #optional trimming
trainFiles = trainFiles[:trim]
testFiles = testFiles[:trim]
trainLabels = trainLabels[:trim]

images = [tf.keras.preprocessing.image.load_img(trainFile) for trainFile in trainFiles]
trainImages = np.array([keras.preprocessing.image.img_to_array(image) for image in images])

images = [tf.keras.preprocessing.image.load_img(testFile) for testFile in testFiles]
testImages = np.array([keras.preprocessing.image.img_to_array(image) for image in images])

#normalize pixel values
trainImages = trainImages / 255.0
testImages = testImages / 255.0

#Model fitting

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

#Uncomment for iteration 1 & 2
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Uncomment for iteration 1
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=20)

#testing

model.predict(testImages)
