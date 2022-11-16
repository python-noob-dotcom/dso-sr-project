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

    def plot_rv_map(self, rv_map_log):
        # rv map parameters
        self.range_axis = self.c * self.fast_time_axis / 2
        slow_time_axis = np.array([x * self.pri for x in any_seq(0, self.num_pulses)])
        dop_axis = time_to_freq(slow_time_axis)
        self.vel_axis = -1 * self.lam * dop_axis / 2

        rv_map_log_squeezed = np.sum(rv_map_log, 2)
        rv_map_log_squeezed_trunc, (xmin, xmax, ymin, ymax) = imagesc_input(rv_map_log_squeezed[:, :], self.vel_axis,
                                                                            self.range_axis / 1e3)

        fig, ax = plt.subplots()
        ax.imshow(rv_map_log_squeezed_trunc, interpolation='none', extent=(xmin, xmax, ymax, ymin))
        ax.set_ylabel('range (km)')
        ax.set_xlabel('velocity (m/s)')

        save_img('rv_map', form='png')

    def process(self, signal_ref, received_signal):
        # reshape data
        data_mat = np.zeros((self.num_rng_bins, self.num_pulses, self.num_array), dtype=complex)
        for x in any_seq(0, self.num_array):
            data_mat[:, :, x] = received_signal[:, x].reshape((self.num_pulses, self.num_rng_bins)).T
        signal_ref_mat = np.tile(signal_ref.reshape(-1, 1), (1, self.num_pulses))

        # matched filter
        logging.info('Match Filtering')
        nfft = 2 * self.num_rng_bins
        data_mf = np.zeros((nfft, self.num_pulses, self.num_array), dtype=complex)
        for x in any_seq(0, self.num_array):
            data_mf[:, :, x] = ifft(fft(data_mat[:, :, x], nfft, 0) * np.conj(fft(signal_ref_mat, nfft, 0)), axis=0)
        data_mf = data_mf[:self.num_rng_bins, :, :]

        # doppler processing
        logging.info('Doppler Processing')
        rv_map_test = np.zeros((self.num_rng_bins, self.num_pulses, self.num_array), dtype=complex)
        rv_map_log = np.zeros((self.num_rng_bins, self.num_pulses, self.num_array))

        for x in any_seq(0, self.num_array):
            rv_map = fftshift(fft(data_mf[:, :, x], axis=1), axes=1)
            rv_map_test[:, :, x] = rv_map
            rv_map_log[:, :, x] = 20 * np.log10(np.abs(rv_map))

        return data_mf, (rv_map, rv_map_log)




