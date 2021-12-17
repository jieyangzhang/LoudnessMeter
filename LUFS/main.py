import numpy as np
import click
from numpy.core.arrayprint import str_format
from scipy.io import wavfile
from tqdm import tqdm

from utils.utils import *
from utils.k_weighting import *
from utils.lufs import *

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

    LoundMeter = LUFS()
    LoundMeter.step(sig)

    print('RMS : %fdB' % RMS(sig))

    return

if __name__ == '__main__':
    main()