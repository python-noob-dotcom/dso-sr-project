from simulator import Signal
import numpy as np
import logging
from numpy import fft, ifft, fftshift, ifftshift
from utils import time_to_freq, any_seq, imagesc_input, save_img

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