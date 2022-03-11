from keras import Sequential
from keras.layers import Dense, Activation, Reshape


def simple_sequential(input_shape, output_shape):

    model = Sequential()
    model.add(Dense(32, input_dim=input_shape[0]))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(output_shape[0]*output_shape[1]))
    model.add(Activation('sigmoid'))

    model.add(Reshape(output_shape))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model
