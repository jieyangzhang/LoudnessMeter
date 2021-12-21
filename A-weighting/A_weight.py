# -*- coding: utf-8 -*-
# @Time    : 21-11-23
# @Author  : zhangjieyang
# @File    : A_weight.py
# @Software: Vscode
import numpy as np
from numpy.core.arrayprint import str_format
from scipy.io import wavfile
import math
import click
from tqdm import tqdm

def RMS(sig):
    rms = 20 * math.log10((np.sum(sig ** 2) / len(sig)) ** 0.5)

    return rms

class Biquad_v1(object):
    def __init__(self, b0, b1, b2, a1, a2):
        self.numerator = np.array([b0, b1, b2]).reshape(1, 3)
        self.denominator = np.array([a1, a2]).reshape(1, 2)
        # state = (y[-1], y[-2], x[0], x[-1], x[-2])
        self.state = np.zeros((5, 1))

        return

    def step(self, x):
        # update x_state
        self.state[4] = self.state[3]
        self.state[3] = self.state[2]
        self.state[2] = x

        # filte : y[0] = x[0] * b[0] + x[-1] * b[1] + x[-2] * b[2] + a[1] * y[-1] + a[2] * y[-2]
        y = np.dot(self.numerator, self.state[2:]) + np.dot(self.denominator, self.state[0:2])

        # update y_state
        self.state[1] = self.state[0]
        self.state[0] = y

        return y

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

class aWeightingFilter(object):
    def __init__(self, Sr):
        if Sr == 16000:
            self.biquad_list = [Biquad_v2(0.4976,  0.9951, 0.4976, -0.8216, -0.1687), \
                                Biquad_v2(0.9302, -1.8604, 0.9302,  1.7055, -0.7160), \
                                Biquad_v2(0.9920, -1.9841, 0.9920,  1.9839, -0.9840)]
            self.scale_values = [1, 1, 1, 1.1575]
        elif Sr == 48000:
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


@click.command()
@click.option(
    '--audio_file',
    prompt='please input audio_file to test',
    type=str_format,
    help='Audio File'
)
def main(audio_file):
    freq, sig = wavfile.read(audio_file)
    sig = sig / (2 ** 15)
    if freq != 16000 | freq != 48000:
        print('input audio_file\'s sample rate should be 16000/48000')
        return
    A_weighting = aWeightingFilter()
    data = sig
    output = np.zeros(len(sig))
    for i in tqdm(range(len(sig))):
        output[i] = A_weighting.step(data[i])
    print('A Loudness : %fdB' % (RMS(output)))
    output *= (2 ** 15)
    wavfile.write("./out.wav", freq, output.astype(np.int16))

    return

if __name__ == '__main__':
    main()