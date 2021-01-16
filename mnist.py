import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train / 255  #归一化
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

'''卷积->relu->pooling'''
model.add(Conv2D(filters=20, kernel_size=(5, 5), input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
'''卷积->relu->pooling'''
model.add(Conv2D(filters=50, kernel_size=(5, 5), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Flatten())  #压平
model.add(Dense(500, activation='relu'))  #全连接
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training....')
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('\nTesting')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)