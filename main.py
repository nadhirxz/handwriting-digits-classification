import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import cv2
import numpy as np

MODEL_SAVED = True
MODEL_NAME = "mnist-model.model"
IMG_SIZE = 28

if MODEL_SAVED:
	print("Loading the model ..")
	model = keras.models.load_model(MODEL_NAME)
else:
	if tf.test.is_gpu_available: print("Using the GPU ...")

	mnist = keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = keras.utils.normalize(x_train, axis=1)
	x_test = keras.utils.normalize(x_test, axis=1)

	model = Sequential()
	model.add(Flatten())
	model.add(Dense(128, activation=tf.nn.relu))
	model.add(Dense(128, activation=tf.nn.relu))
	model.add(Dense(10, activation=tf.nn.softmax))

	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	print("Training ..")
	model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

	model.save(MODEL_NAME)

while True:
	img = input("Your image : ")
	img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	predictions = model.predict(np.array([img]))

	print(np.argmax(predictions))
