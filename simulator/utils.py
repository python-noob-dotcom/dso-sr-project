import os
import numpy as np
import matplotlib.pyplot as plt
import logging

LOG = logging.getLogger(__name__)




def time_to_freq(tvec):
    '''convert time vector to frequency vector for `fft`
       generate the frequency axis `fvec` from preiodically sampled time axis `tvec`
       it is useful for `fft` followed by `fftshift`

       input:
           fvec: spans from -fs/2 to fs/2
       output:
           tvec: applicable to both odd and even axis length
    '''

    n = len(tvec)
    fs = (n - 1) / (tvec[-1] - tvec[0])

    if n % 2 == 1:  # odd
        fvec = np.array([x * fs / n for x in any_seq(int(-(n - 1) / 2), int((n - 1) / 2) + 1)])
    else:  # even
        fvec = np.array([x * fs / n for x in any_seq(int(-n / 2), int(n / 2 - 1) + 1)])
    return fvec


def freq_to_time(self, fvec):
    '''convert freq vector to time vector for `ifft`
       generate the periodically-sampled time axis `tvec` from frequency axis `fvec`
       it is useful for `ifftshift` followed by `ifft`

       input:
           fvec: spans from -fs/2 to fs/2
       output:
           tvec: spans from 0 to t*(n-1)
    '''

    n = len(fvec)
    t = (n - 1) / (fvec[-1] - fvec[0])
    tvec = np.array([x * t / n for x in any_seq(0, n)])
    return tvec


def next_pow_two(x):
    '''returns the exponents for smallest power of two that is greater than or equal to non-negative integer x'''

    x = int(x)
    return 0 if x == 0 else (x - 1).bit_length()


def any_seq(start, stop, step=1):
    '''generates a sequence of numbers from start (inclusive) to end (exclusive) of specified step (default: 1)'''

    assert step >= 0, 'step should be non-negative'

    if start <= stop:
        seq = [x for x in range(start, stop, step)]
    else:
        seq = [x for x in range(start, stop, -1 * step)]
    return seq


def imagesc_input(matrix, x, y):
    xmin, xmax, ymin, ymax = x[0], x[-1], y[0], y[-1]

    if xmin > xmax:
        xmin, xmax = xmax, xmin
        matrix = np.flip(matrix, axis=1)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
        matrix = np.flip(matrix, axis=0)

    return matrix, (xmin, xmax, ymin, ymax)


def save_img(name, folder='result', form='png'):
    savepath = '{}/{}.{}'.format(folder, name, form)
    plt.savefig(savepath)
    logging.info('Output image is saved to {}'.format(savepath))


def make_dir(folder='result'):
    if not os.path.exists(folder):
        os.mkdir(folder)
