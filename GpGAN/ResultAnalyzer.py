import matplotlib.pyplot as plt
from keras.utils import plot_model


class ResultAnalyzer:
    def __init__(self):
        pass

    # Plot the loss from each batch
    def plotLoss(self, epoch, gLosses, dLosses):
        plt.figure(figsize=(10, 8))
        plt.plot(dLosses, label='Discriminitive loss')
        plt.plot(gLosses, label="Generative loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

    def plotAccuracy(self,history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def saveModels(self, generator, discriminator, generator_path, discriminator_path):
        generator.save(generator_path)
        discriminator.save(discriminator_path)

    def visualizeModelGraph(self,model,filename):
        model.summary()
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)


