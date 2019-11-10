import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


class ModelPrediction:
    def __init__(self):
        pass

    def plotGeneratedImages(self, epoch, modelpath, examples=100, dim=(10, 10), figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, 100])
        generator = load_model(modelpath)
        generatedImages = generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generatedImages[i, 0], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def generateOneGanStimuli(self, epoch, modelpath, examples=100, figsize=(5, 5)):
        noise = np.random.normal(0, 1, size=[examples, 100])
        generator = load_model(modelpath)
        generatedImages = generator.predict(noise)
        plt.figure(figsize=figsize)

        plt.imshow(generatedImages[1, 0], interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


model_prediction = ModelPrediction()
epochs = 100
model_path = '/home/aswin/PycharmProjects/GpGAN/models/dcgan_bargenerator_epoch_100.h5'
model_prediction.generateOneGanStimuli(epochs, model_path)
model_prediction.plotGeneratedImages(epochs, model_path)
