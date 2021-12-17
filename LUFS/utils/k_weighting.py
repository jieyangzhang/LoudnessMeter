import numpy as np
from numpy.core.arrayprint import str_format
from scipy.io import wavfile
from tqdm import tqdm

# 二阶直接II型IIR滤波器
class Biquad_v2(object):
    def __init__(self, b0, b1, b2, a1, a2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        # stete[2] = [d1, d2]
        self.state = np.zeros(2)

        return

    def step(self, x):
        y = self.b0 * x + self.state[0]
        self.state[0] = self.b1 * x + self.a1 * y + self.state[1]
        self.state[1] = self.b2 * x + self.a2 * y

        return y

# K计权滤波
class kWeightingFilter(object):
    def __init__(self):
        self.biquad_list = [Biquad_v2(1.53512485958697, -2.69169618940638, 1.19839281085285, 1.69065929318241, -0.73248077421585), \
                            Biquad_v2(1.0, -2, 1, 1.99004745483398, -0.99007225036621)]
        self.scale_values = [1, 1, 1]
        self.cascade_size = 2

        return

    def step(self, x):
        y = x
        for i in range(self.cascade_size):
            y = self.biquad_list[i].step(y)

        return y

class aWeightingFilter(object):
    def __init__(self):
        self.biquad_list = [Biquad_v2(1,  2, 1, 0.2246, -0.0126), \
                            Biquad_v2(1, -2, 1, 1.8939, -0.8952), \
                            Biquad_v2(1, -2, 1, 1.9946, -0.9946)]
        self.scale_values = [0.2343, 1, 1, 1]
        self.cascade_size = 3

        return

    def step(self, x):
        y = x
        for i in range(self.cascade_size):
            y = self.biquad_list[i].step(y * self.scale_values[i])
        y = y * self.scale_values[-1]

        return y
