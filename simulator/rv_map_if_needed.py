from simulator import Signal
from utils import any_seq, time_to_freq, save_img, imagesc_input
import numpy as np
import logging
import matplotlib.pyplot as plt


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