import math
import numpy as np
import sys
sys.path.append('../')
from GpGAN.Perception.util import Util

class Figure1:

  def __init__(self):
    pass

  DELTA_MIN = 20
  DELTA_MAX = 80
  SIZE = (100, 100)

  def length(self,flags=[False, False, False], preset=None):

    var_y = flags[0]
    var_x = flags[1]
    var_width = flags[2]


    sparse = None
    image = None
    label = None
    parameters = 1

    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    LENGTH, p = Util.parameter(1, Y_RANGE[1]-Y_RANGE[0]+1) # 1..60
    parameters *= p

    if preset:
      LENGTH = preset

    MAX_LENGTH = Y_RANGE[1]-Y_RANGE[0]
    # print 'Max length', MAX_LENGTH

    X = math.floor(Figure1.SIZE[1] / 2)
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Y_RANGE[0]
    if var_y:
      
      Y, p = Util.parameter(0, Figure1.SIZE[0]-MAX_LENGTH)
      # print 'Y',Y
      parameters *= p

    WIDTH = 1
    if var_width:
      sizes = [1, 3, 5, 7, 9, 11]
      WIDTH = np.random.choice(sizes)
      parameters *= len(sizes)

    sparse = [Y, X, LENGTH, WIDTH]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)


    half_width = math.floor(WIDTH / 2) # this always floors
    
    # print(Y,LENGTH,X,half_width,WIDTH)
    image[Y:Y+LENGTH, X-half_width:X+half_width+1] = 1


    label = LENGTH

    return sparse, image, label, parameters



  def position_common_scale(flags=[False, False], preset=None):
    '''
    '''
    var_x = flags[0]
    var_spot = flags[1]


    sparse = None
    image = None
    label = None
    parameters = 1


    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
    parameters *= p

    if preset:
      Y = preset

    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    SPOT_SIZE = 5
    if var_spot:
      sizes = [1, 3, 5, 7, 9, 11]
      SPOT_SIZE = np.random.choice(sizes)
      parameters *= len(sizes)

    ORIGIN = 10

    sparse = [Y, X, SPOT_SIZE]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw axis
    image[Y_RANGE[0]:Y_RANGE[1], ORIGIN] = 1

    # draw spot
    half_spot_size = SPOT_SIZE / 2
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = Y - Figure1.DELTA_MIN

    return sparse, image, label, parameters


