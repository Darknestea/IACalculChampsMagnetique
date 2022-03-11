import math

from keras import Sequential
from keras.initializers.initializers_v2 import HeNormal
from keras.layers import Dense, Activation, Reshape, UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout


def simple_deconvolution(input_shape, output_shape, name):
    image_width = output_shape[0]

    initializer = HeNormal()

    # Check image width is power of two
    assert (image_width - 2**int(math.log2(image_width)) < 0.0001)

    depth = int(image_width / 8)
    dropout = 0.9

    model = Sequential(name=name)
    model.add(Dense(100, input_shape=(input_shape[0],), kernel_initializer=initializer))
    model.add(Activation('relu'))

    model.add(Dense(256, kernel_initializer=initializer))
    model.add(Activation('relu'))

    model.add(Dense(image_width * 8))
    # model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Reshape((8, 8, depth)))
    # model.add(Dropout(dropout))

    while depth > 2:
        depth = int(depth/2)
        model.add(UpSampling2D())

        model.add(Conv2DTranspose(depth, 3, padding='same'))
        # model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

    model.add(UpSampling2D())
    # model.add(BatchNormalization(momentum=0.9))
    model.add(Conv2DTranspose(1, 3, padding='same'))
    model.add(Activation('sigmoid'))

    # Out: grayscale image [0.0,1.0] per pix
    model.add(Reshape(output_shape))

    model.compile(optimizer='sgd', loss='mean_squared_error')

    return model
