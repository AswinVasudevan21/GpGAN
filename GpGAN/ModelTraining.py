from tqdm import tqdm
import numpy as np
from keras.models import Model, Sequential

from GpGAN.ResultAnalyzer import ResultAnalyzer


class ModelTraining:

    def __init__(self):
        pass

    def trainGAN(self,epochs=1, batchSize=128,gan=Model(),X_train=(),generator=Sequential(),discriminator=Sequential(), generatorpath='', discriminatorpath=''):
        gLosses = []
        dLosses = []
        result_analyzer = ResultAnalyzer()
        batchCount = X_train.shape[0] / batchSize
        xrange=range
        for e in xrange(1, epochs + 1):
            print('-' * 15, 'Epoch %d' % e, '-' * 15)
            for _ in tqdm(xrange(int(batchCount))):
                noise = np.random.normal(0, 1, size=[batchSize, 100])
                imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

                # Generate fake stimuli images
                generatedImages = generator.predict(noise)
                X = np.concatenate([imageBatch, generatedImages])

                # Dummy labels for GAN stimuli
                yDis = np.zeros(2 * batchSize)
                yDis[:batchSize] = 0.9

                discriminator.trainable = True
                dloss = discriminator.train_on_batch(X, yDis)
                noise = np.random.normal(0, 1, size=[batchSize, 100])
                yGen = np.ones(batchSize)
                discriminator.trainable = False
                gloss = gan.train_on_batch(noise, yGen)

            dLosses.append(dloss)
            gLosses.append(gloss)

            if e == 1 or e % 5 == 0:
                result_analyzer.saveModels(generator,discriminator,generatorpath,discriminatorpath)










