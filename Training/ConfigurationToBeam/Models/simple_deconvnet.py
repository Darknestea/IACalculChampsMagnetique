from keras import Sequential
from keras.layers import Dense, Activation, Reshape, BatchNormalization, Dropout, UpSampling2D, Conv2DTranspose


def simple_deconvolution(input_shape, output_shape):
    dim = int(input_shape[0] / 8)
    depth = input_shape[0] * 2
    dropout = 0.9

    model = Sequential()
    model.add(Dense(32, input_shape=(input_shape[0],)))
    model.add(Activation('relu'))

    model.add(Dense(output_shape[0]))
    model.add(Activation('relu'))

    model.add(Dense(dim * dim * depth))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(1, 5, padding='same'))
    model.add(Activation('sigmoid'))

    # Out: grayscale image [0.0,1.0] per pix
    model.add(Reshape(output_shape))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    return model
