#DenseNet architecture for a 30x30 image
model = Sequential()

# 1st convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(30, 30, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Dense block 1
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Concatenate())

# Dense block 2
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Concatenate())

# Dense block 3
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Concatenate())

# global pooling
