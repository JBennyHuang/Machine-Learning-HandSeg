import tensorflow as tf
import tensorflow.keras as keras

def unet(x, num_classes):
    net = keras.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    skip_1 = net
    net = keras.layers.MaxPool2D((2, 2), (2, 2), 'same')(net)

    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    skip_2 = net
    net = keras.layers.MaxPool2D((2, 2), (2, 2), 'same')(net)

    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    skip_3 = net
    net = keras.layers.MaxPool2D((2, 2), (2, 2), 'same')(net)

    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(512, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.UpSampling2D((2,2))(net)
    net = tf.concat((net, skip_3), 3)

    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.UpSampling2D((2,2))(net)
    net = tf.concat((net, skip_2), 3)

    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.UpSampling2D((2,2))(net)
    net = tf.concat((net, skip_1), 3)

    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)

    net = keras.layers.Conv2D(num_classes, (1, 1), (1, 1), 'same')(net)

    return net