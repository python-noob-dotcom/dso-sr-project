import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift
import random
from FCN import FCN
from utils import time_to_freq, any_seq, imagesc_input, save_img

LOG = logging.getLogger(__name__)


class Signal(object):

    def __init__(self):
        super(Signal, self).__init__()

        # constants
        self.c = 3e8  # speed of light

        # signal parameters
        self.fs = 4e9  # carrier frequency, Hz
        self.b = 0.8 * self.fs  # bandwidth, Hz
        self.fc = 2 * self.b  # sampling rate, Hz
        self.lam = self.c / self.fc  # wavelength, m

        # pulse and signal processing parameters
        self.num_rng_bins = 2000  # to determine the PRI. for simplicity of simulation
        self.pri = self.num_rng_bins / self.fs
        self.num_pulses = 400
        self.pw = 50 / self.fs  # pulse width, s
        self.total_number_samples = self.num_rng_bins * self.num_pulses  # data time samples
        self.cit = self.total_number_samples / self.fs  # coherent integration time

        # array parameters
        self.num_array = 50
        self.spacing_array = 0.1

        # target parameters
        self.tgt_snr_db = np.array([])  # dB
        self.tgt_range = np.array([])  # m
        self.tgt_velocity = np.array([])
        self.angle_of_arrival_deg = np.array([])
        for i in range(0,10):
            self.tgt_snr_db = np.append(self.tgt_snr_db, random.randint(0, 100, 10))
            self.tgt_range = np.append(self.tgt_range, random.randint(100000, 1000000, 10000))
            self.tgt_velocity = np.append(self.tgt_velocity, random.randint(100, 300, 20))
            self.angle_of_arrival_deg = np.append(self.angle_of_arrival_deg, random.randint(10, 100, 5))
        self.angle_of_arrival_rad = (self.angle_of_arrival_deg / 360) * 2 * np.pi
        self.tgt_num = len(self.tgt_range)  # counting number of targets

        self.fcn = FCN()

    def simulate(self):
        # reference signal simulation
        self.fast_time_axis = np.array([x / self.fs for x in any_seq(0, self.num_rng_bins)])
        signal_ref = self.fcn.get_chirp(self.fast_time_axis, self.b, self.pw)

        # target signal simulation
        t0 = 2 * self.tgt_range / self.c
        tgt_doppler_hz = -2 * self.tgt_velocity / self.lam
        full_time_axis = np.array([x / self.fs for x in any_seq(0, self.total_number_samples)])

        # simulate more than one target
        signal_target = np.zeros((self.num_array, self.total_number_samples, self.tgt_num), dtype=complex)
        for target_num in any_seq(0, self.tgt_num):
            logging.info('Simulating Target {} ...'.format(target_num))
            for element in any_seq(0, self.num_array):
                steering_vector = np.exp(-1j * 2 * np.pi * element * self.spacing_array * \
                                         np.sin(self.angle_of_arrival_rad[target_num]) / self.lam)
                signal_target[element, :, target_num] = np.exp(
                    1j * 2 * np.pi * tgt_doppler_hz[target_num] * full_time_axis) * self.fcn.get_chirp_train(
                    full_time_axis, self.b, self.pw, 0, t0[target_num], self.pri) * steering_vector

        # combine all signals and noise into one vector
        received_signal = np.zeros((self.total_number_samples, self.num_array), dtype=complex)
        logging.info('Combining Signals & Adding Noise')
        for x in any_seq(0, self.total_number_samples):
            for y in any_seq(0, self.num_array):
                noise = 1 / np.sqrt(2) * (np.random.normal() + 1j * np.random.normal())
                received_signal[x, y] = np.sum(signal_target[y, x, :], 0) + noise

        return signal_ref, received_signal


