[![build passing](https://travis-ci.org/ukubuka/ukubuka-core.svg?branch=master	)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![python version](	https://img.shields.io/badge/Python-3.6-blue.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![tensorflow version](	https://img.shields.io/badge/Tensorflow-1.2-yellow.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)
[![keras version](	https://img.shields.io/badge/Keras-2.6-green.svg)](https://github.com/AswinVasudevan21/HandGestureClassification/blob/master/README.md)




# GpGAN - Graphical perception based GAN
### Objective
The goal of this project is to generate stimuli using generator and generalize the discriminator to predict the variations in translation and stroke of stimuli using random noise.

### Steps in Execution:
To Train model from scratch

    1. Clone or Download the project
    2. Pip install the requirements.txt
    3. Set epochs,batch size, model path location
    4. Run ModelPipeline.py on python 3


To Predict trained model:

    1. Repeat the first three steps as above.
    2. Set model path location
    2. Run ModelPrediction.py on python 3

### Architecture:
Graphical perception technique involves visual decoding of qualitative and quantitative information from the graphs as demonstrated by Cleveland and Mcgill[[1]]. This work is based on Evaluating Graphical perception with CNN where we have experimented the hypothesis (H1.4)[[2]] with GAN. The GAN network [[3]] is treated as deeply convoluted network where generator produces newer representation of visual stimuli and discriminator model would eventually generalize to newer added variations in stimuli. 

### Visualization:

#### GAN Generated Stimuli
<img height="300px" src="https://github.com/AswinVasudevan21/GpGAN/blob/master/images/ganstimuli.png">

#### Random Noise Stimuli
<img height="300px" src="https://github.com/AswinVasudevan21/GpGAN/blob/master/images/noiseimage.png">

### Future Work:
Our next objective is to make discriminator model generalize to predict the random noise added stimuli.
Fine tune model with hyper parameters 

### References:
[[1]]: http://euclid.psych.yorku.ca/www/psy6135/papers/ClevelandMcGill1984.pdf <br>
[[2]]: https://danielhaehn.com/papers/haehn2018evaluating.pdf <br>
[[3]]: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf <br>
