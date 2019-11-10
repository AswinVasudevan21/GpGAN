import time
import numpy as np
import os
import matplotlib.pyplot as plt

from GpGAN.DataAugmentation import DataAugmentation
from GpGAN.ResultAnalyzer import ResultAnalyzer

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.optimizers import Adam
from keras import backend as K

from GpGAN.GanModelCreator import GanModelCreator
from GpGAN.Perception.figure1 import Figure1
from GpGAN.Perception.figure12 import Figure12
from GpGAN.ModelTraining import ModelTraining



class ModelPipeline:
    def __init__(self):
        pass

    def prepareBarXandY(self):
        EXPERIMENT = "Figure12.data_to_framed_rectangles"
        DATATYPE = eval(EXPERIMENT)
        train_counter = 0
        val_counter = 0
        test_counter = 0
        train_target = 6000
        val_target = 2000
        test_target = 2000
        NOISE=False

        train_labels = []
        val_labels = []
        test_labels = []

        X_train = np.zeros((train_target, 100, 100), dtype=np.float32)
        y_train = np.zeros((train_target, 2), dtype=np.float32)

        X_val = np.zeros((val_target, 100, 100), dtype=np.float32)
        y_val = np.zeros((val_target, 2), dtype=np.float32)

        X_test = np.zeros((test_target, 100, 100), dtype=np.float32)
        y_test = np.zeros((test_target, 2), dtype=np.float32)

        t0 = time.time()

        all_counter = 0
        while train_counter < train_target or val_counter < val_target or test_counter < test_target:

            all_counter += 1

            data, label, parameters = Figure12.generate_datapoint()

            pot = np.random.choice(3)

            # sometimes we know which pot is right
            if label in train_labels:
                pot = 0
            if label in val_labels:
                pot = 1
            if label in test_labels:
                pot = 2

            if pot == 0 and train_counter < train_target:

                if label not in train_labels:
                    train_labels.append(label)

                #
                image = DATATYPE(data)
                image = image.astype(np.float32)

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to training
                X_train[train_counter] = image
                y_train[train_counter] = label
                train_counter += 1

            elif pot == 1 and val_counter < val_target:

                if label not in val_labels:
                    val_labels.append(label)

                image = DATATYPE(data)
                image = image.astype(np.float32)

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to training
                X_val[val_counter] = image
                y_val[val_counter] = label
                val_counter += 1

            elif pot == 2 and test_counter < test_target:

                if label not in test_labels:
                    test_labels.append(label)

                image = DATATYPE(data)
                image = image.astype(np.float32)

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to training
                X_test[test_counter] = image
                y_test[test_counter] = label
                test_counter += 1

        print
        'Done', time.time() - t0, 'seconds (', all_counter, 'iterations)'
        #
        #
        #

        #
        #
        # NORMALIZE DATA IN-PLACE (BUT SEPERATELY)
        #
        #
        X_min = X_train.min()
        X_max = X_train.max()
        y_min = y_train.min()
        y_max = y_train.max()

        # scale in place
        X_train -= X_min
        X_train /= (X_max - X_min)
        y_train -= y_min
        y_train /= (y_max - y_min)

        X_val -= X_min
        X_val /= (X_max - X_min)
        y_val -= y_min
        y_val /= (y_max - y_min)

        X_test -= X_min
        X_test /= (X_max - X_min)
        y_test -= y_min
        y_test /= (y_max - y_min)

        # normalize to -.5 .. .5
        X_train -= .5
        X_val -= .5
        X_test -= .5

        return X_train, y_train, X_test, y_test

    def prepareXandY(self):
        train_target = 600
        val_target = 200
        test_target = 200
        NOISE = False

        # get global min and max
        global_min = np.inf
        global_max = -np.inf
        for N in range(train_target + val_target + test_target):
            sparse, image, label, parameters = Figure1.length([True, True, True])

            global_min = min(label, global_min)
            global_max = max(label, global_max)
        # end of global min max

        X_train = np.zeros((train_target, 100, 100), dtype=np.float32)
        y_train = np.zeros((train_target), dtype=np.float32)
        train_counter = 0

        X_val = np.zeros((val_target, 100, 100), dtype=np.float32)
        y_val = np.zeros((val_target), dtype=np.float32)
        val_counter = 0

        X_test = np.zeros((test_target, 100, 100), dtype=np.float32)
        y_test = np.zeros((test_target), dtype=np.float32)
        test_counter = 0

        t0 = time.time()

        min_label = np.inf
        max_label = -np.inf

        all_counter = 0
        while train_counter < train_target or val_counter < val_target or test_counter < test_target:

            all_counter += 1

            sparse, image, label, parameters = Figure1.length([True, True, True])

            # we need float
            image = image.astype(np.float32)

            pot = np.random.choice(3)  # , p=([.6,.2,.2]))

            if label == global_min or label == global_max:
                pot = 0  # for sure training

            if pot == 0 and train_counter < train_target:
                # a training candidate
                if label in y_val or label in y_test:
                    # no thank you
                    continue

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to training
                X_train[train_counter] = image
                y_train[train_counter] = label
                train_counter += 1

            elif pot == 1 and val_counter < val_target:
                # a validation candidate
                if label in y_train or label in y_test:
                    # no thank you
                    continue

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to validation
                X_val[val_counter] = image
                y_val[val_counter] = label
                val_counter += 1

            elif pot == 2 and test_counter < test_target:
                # a test candidate
                if label in y_train or label in y_val:
                    # no thank you
                    continue

                # add noise?
                if NOISE:
                    image += np.random.uniform(0, 0.05, (100, 100))

                # safe to add to test
                X_test[test_counter] = image
                y_test[test_counter] = label
                test_counter += 1

        print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')

        X_min = X_train.min()
        X_max = X_train.max()
        y_min = y_train.min()
        y_max = y_train.max()

        # scale in place
        X_train -= X_min
        X_train /= (X_max - X_min)
        y_train -= y_min
        y_train /= (y_max - y_min)

        X_val -= X_min
        X_val /= (X_max - X_min)
        y_val -= y_min
        y_val /= (y_max - y_min)

        X_test -= X_min
        X_test /= (X_max - X_min)
        y_test -= y_min
        y_test /= (y_max - y_min)

        # normalize to -.5 .. .5
        X_train -= .5
        X_val -= .5
        X_test -= .5

        print('memory usage', (X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1000000.,'MB')

        return X_train,y_train,X_test,y_test


#setting up parameters
model_pipeline = ModelPipeline()
X_train,y_train,X_test,y_test=model_pipeline.prepareBarXandY()
real_X = X_train
#X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]
print("Prepared the dataset:"+str(X_train.shape))

#prediction noise image
data_augmentation = DataAugmentation()
data_augmentation.noiseImage(real_X[100])

#setting up GAN
K.set_image_dim_ordering('th')
np.random.seed(1000)
randomDim = 100
adam = Adam(lr=0.0002, beta_1=0.5)
gan_model = GanModelCreator()
generator = gan_model.setGenerator(adam,randomDim)
discriminator = gan_model.setDiscriminator(adam,randomDim)
gan= gan_model.connectGeneratorDiscriminator(discriminator,generator,adam,randomDim)
print("Completed GAN setup")

#Model Training
model_training = ModelTraining()
epochs=100
batchsize = 256
generator_path = '/home/aswin/PycharmProjects/GpGAN/models/dcgan_bargenerator_epoch_'+str(epochs)+'.h5'
discriminator_path = '/home/aswin/PycharmProjects/GpGAN/models/dcgan_bardiscriminator_epoch_'+str(epochs)+'.h5'
print("Model training started")
model_training.trainGAN(epochs,batchsize,gan,X_train,generator,discriminator, generator_path,discriminator_path)
print("Completed Model Training")




