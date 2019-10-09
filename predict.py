import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2 as cv2

from glob import glob
from unet import unet

tf.executing_eagerly()

capture = cv2.VideoCapture(0)

model = keras.models.load_model('models/model.h5')

while True:
    ret, img = capture.read()

    y_pred = model.predict(np.array([cv2.resize(img, (256, 256))]).astype(float))

    cv2.imshow('', np.sum(y_pred[0,...,1:], axis=-1).round())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break