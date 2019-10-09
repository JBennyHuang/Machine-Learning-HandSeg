import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2 as cv2

from glob import glob
from architectures.unet import unet


# tf.executing_eagerly()

class Dataset:
    def read_image(self, x, y):
        x = tf.io.decode_png(tf.io.read_file(x), channels=3)
        y = tf.io.decode_png(tf.io.read_file(y), channels=1)

        return x, y

    def process_image(self, x, y):
        x = tf.cast(x, tf.float32)

        y = tf.stack([tf.logical_or(tf.equal(tf.squeeze(y), 0), tf.equal(tf.squeeze(y), 1))] + \
            [tf.equal(tf.squeeze(y), c) for c in range(2, 34)], axis=-1)
        y = tf.cast(y, tf.float32)

        stack = tf.concat([x, y], axis=2)
        stack = tf.image.random_crop(stack, [128, 128, 36])
        stack = tf.image.random_flip_left_right(stack)
        stack = tf.image.random_flip_up_down(stack)
        stack = tf.image.rot90(stack , tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        x = stack[:,:,:3]
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)

        y = stack[:,:,3:]

        return x, y

    def __init__(self, images_path, labels_path, batch_size):
        images_filenames = glob(f'{images_path}/*.png')
        labels_filenames = glob(f'{labels_path}/*.png')
        
        filenames = (images_filenames, labels_filenames)
        
        self._dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self._dataset = self._dataset.shuffle(len(filenames[0]))
        self._dataset = self._dataset.prefetch(batch_size)
        self._dataset = self._dataset.map(self.read_image, 4)
        self._dataset = self._dataset.map(self.process_image, 4)
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.batch(batch_size)

    def __call__(self):
        return self._dataset

images_path = '..'
labels_path = '..'

d = Dataset(images_path, labels_path, 32)

def categorical_crossentropy(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 33))
    y_pred = tf.reshape(y_pred, (-1, 33))

    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)


# inputs = keras.Input(shape=(None, None, 3))
# outputs = tf.nn.softmax(unet(inputs, 33))

# model = keras.Model(inputs=inputs, outputs=outputs)

# model.compile(loss=categorical_crossentropy,
#             optimizer=keras.optimizers.Adam(),
#             metrics=['accuracy'])

model = keras.models.load_model('model.h5')

model.fit(d(),
            steps_per_epoch=100, 
            epochs=50)

model.save('model.h5')