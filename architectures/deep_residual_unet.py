import tensorflow as tf
import tensorflow.keras as keras

# https://arxiv.org/pdf/1711.10684.pdf
def deep_residual_unet(x, num_classes):
    # Encoder Block 1
    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)

    res = keras.layers.Conv2D(64, (1, 1), (2, 2), 'same')(x)

    net = tf.math.add(net, res)

    skip_1 = net

    # Encoder Block 2
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(128, (3, 3), (2, 2), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)

    res = keras.layers.Conv2D(128, (1, 1), (2, 2), 'same')(skip_1)

    net = tf.math.add(net, res)

    skip_2 = net

    # Encoder Block 3
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(256, (3, 3), (2, 2), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)

    res = keras.layers.Conv2D(256, (1, 1), (2, 2), 'same')(skip_2)
    
    net = tf.math.add(net, res)

    skip_3 = net

    # Bridge
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(512, (3, 3), (2, 2), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(512, (3, 3), (1, 1), 'same')(net)

    res = keras.layers.Conv2D(512, (1, 1), (2, 2), 'same')(skip_3)

    net = tf.math.add(net, res)

    # Decoder Block 3
    net = keras.layers.UpSampling2D((2,2))(net)

    res = tf.concat([net, skip_3])

    net = keras.layers.BatchNormalization()(res)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(256, (3, 3), (1, 1), 'same')(net)

    net = tf.math.add(net, res)

    # Decoder Block 2
    net = keras.layers.UpSampling2D((2,2))(net)

    res = tf.concat([net, skip_2])

    net = keras.layers.BatchNormalization()(res)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(128, (3, 3), (1, 1), 'same')(net)

    net = tf.math.add(net, res)

    # Decoder Block 1
    net = keras.layers.UpSampling2D((2,2))(net)

    res = tf.concat([net, skip_1])

    net = keras.layers.BatchNormalization()(res)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Activation('relu')(net)
    net = keras.layers.Conv2D(64, (3, 3), (1, 1), 'same')(net)

    net = tf.math.add(net, res)

    # Output
    net = keras.layers.Conv2D(num_classes, (1, 1), (1, 1), 'same')(net)

    return net


