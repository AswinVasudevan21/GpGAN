from GpGAN.Perception.figure1 import Figure1
import matplotlib.pyplot as plt
import matplotlib

class DataGenerator:
    def __init__(self):
        pass

    def displayImage(self):
        sparse, image, label, parameters = Figure1.length([True, True, False])
        print ('Label', label, 'Parameters', parameters)
        plt.imshow(image)
        plt.show()

    def saveImage(self,images,labels):
        dataset_path = '/home/aswin/PycharmProjects/GpGAN/Dataset/'
        for i in range(0, len(images)):
            print("image creation attempt:" + str(i))
            image_name = dataset_path + 'length' + str(labels[i]) + '.png'
            matplotlib.image.imsave(image_name, images[i])

    def generatePieChart(self):

        labels = ['A', 'B', 'C', 'D']
        sizes = [10, 20, 30, 40]
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        plt.figure(figsize=(1, 1))
        patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

        plt.axis('equal')
        plt.tight_layout()

        plt.savefig("pie101.png")
        plt.show()

datagen = DataGenerator()
datagen.generatePieChart()



