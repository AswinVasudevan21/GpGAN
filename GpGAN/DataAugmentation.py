import numpy as np
import matplotlib.pyplot as plt
from cv2 import warpAffine
from skimage.util import random_noise
from skimage.transform import rotate


class DataAugmentation:
    def __init__(self):
        pass

    def showImage(self, title, image, formatimage):
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle(title, fontsize=16)
        ax1.imshow(image)
        ax1.set_title("Original")
        ax2.imshow(formatimage)
        ax2.set_title("Augmented")
        plt.show()

    def rotateImage(self, imgarray):
        rotate_img = rotate(imgarray, 15)
        self.showImage("Image vs RotateImage", imgarray, rotate_img)

    def translateImage(self, imgarray):
        rows =224
        cols =224
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        translate_img = warpAffine(imgarray, M, (cols, rows))

        self.showImage("Image vs TranslationImage", imgarray, translate_img)

    def noiseImage(self, imgarray):
        noise_img = random_noise(imgarray)
        self.showImage("Image vs NoiseImage", imgarray, noise_img)
