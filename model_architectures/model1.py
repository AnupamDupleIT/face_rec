# ResNet-50 architecture for a 30x30 image classification 
model = Sequential()

# 1st convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(30, 30, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 2nd convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Residual block 1
model.add(Residual(64))
model.add(Residual(64))

# Residual block 2
model.add(Residual(128, 2))
model.add(Residual(128))

# Residual block 3
model.add(Residual(256, 2))
model.add(Residual(256))

# Residual block 4
model.add(Residual(512, 2))
model.add(Residual(512))

# global pooling and fully connected layer
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='softmax'))
