import tensorflow as tf
from tensorflow import keras

import pandas as pd
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

trainLabelsCSV = pd.read_csv('./histopathologic-cancer-detection-data/train_labels.csv')
trainLabels = trainLabelsCSV['label']

trainFiles = glob('./histopathologic-cancer-detection-data/train/*.tif')

counts, bins = np.histogram(trainLabels)
#plt.hist(bins[:-1], bins, weights=counts)
#plt.show()

trim = 50 #optional trimming
trainFiles = trainFiles[:trim]

images = [tf.keras.preprocessing.image.load_img(trainFile) for trainFile in trainFiles]
trainImages = np.array([keras.preprocessing.image.img_to_array(image) for image in images])

trainImages = np.transpose(trainImages)
R,G,B = trainImages[0],trainImages[1],trainImages[2]


counts, bins = np.histogram(B.flatten())
plt.hist(bins[:-1], bins, weights=counts)
plt.show()
