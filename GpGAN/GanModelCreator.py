import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras import initializers


class GanModelCreator:
    def __init__(self):
        pass

    def setGenerator(self, adam, randomDim):
        generator = Sequential()
        generator.add(Dense(128 * 25 * 25, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Reshape((128, 25, 25)))
        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        generator.add(LeakyReLU(0.2))
        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=adam)
        return generator

    def setDiscriminator(self, adam, randomDim):
        discriminator = Sequential()
        discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 100, 100),
                                 kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=adam)
        return discriminator

    def connectGeneratorDiscriminator(self, discriminator, generator, adam, randomDim):
        discriminator.trainable = False
        ganInput = Input(shape=(randomDim,))
        x = generator(ganInput)
        ganOutput = discriminator(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer=adam)
        return gan
