import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

from utils import time_to_freq, next_pow_two, any_seq


class FCN(object):
    def __init__(self):
        super(FCN, self).__init__()
        # self.lam = None                   ## TODO
        # self.dD = None
        # self.dR = None

    def get_chirp(self, t_axis, b, pw, f0=0, t0=None):
        '''generates a upchirp pulse
           for down chirp, set B to negative value

           inputs:
               t_axis : time axis, seconds (size [1 x sig_len])
               b      : bandwidth, Hz (can be negative)
               pw     : pulse width, seconds
               f0     : [optional] center frequency, Hz
                        default: f0=0
               t0     : [optional] puulse start time (according to time in t_axis), seconds
                        default: t0 = t_axis[0]
            output:
               signal : size [1 x sig_len]
        '''

        if t0 == None:
            t0 = t_axis[0]

        cond = (t_axis - t0 >= 0) & (t_axis - t0 <= pw)
        signal = np.exp(1j * 2 * np.pi * f0 * (t_axis - t0 - pw / 2)) * np.exp(
            1j * np.pi * b / pw * (t_axis - t0 - pw / 2) ** 2) * cond

        return signal

    def get_chirp_train(self, t_axis, b, pw, f0, t0, pri):
        '''generates a upchirp pulse train
           for down chirp, set B to negative value

           inputs:
               t_axis : time axis, seconds (size [1 x sig_len])
               b      : bandwidth, Hz (can be negative)
               pw     : pulse width, seconds
               f0     : [optional] center frequency, Hz
                        default: f0=0
               t0     : [optional] puulse start time (according to time in t_axis), seconds
                        default: t0 = t_axis[0]
               pr     : pulse repetition interval, seconds
            output:
               signal : size [1 x sig_len]
        '''

        signal = np.zeros(t_axis.shape)
        num_pulse = np.ceil((t_axis[-1] - t0) / pri)

        for pulse_ind in any_seq(0, int(num_pulse)):
            signal = signal + self.get_chirp(t_axis, b, pw, f0, t0 + pulse_ind * pri)

        return signal

    def get_target_signal_simple(self, signal_ref, range_bin, vel_bin):
        '''computes the idealised signal returns from a target
           this assumes no range migration effects even at high velocities

           inputs:
               signal ref : reference signal (size [1 x len*num_len] or [1 x n]), but will
                            not fall exactly on grid for integer valued `range_bin` and `vel_bin`
               range_bin  : number of range bins delayed, 0 means no delay
                            can take non-integer values
                            can take negative values
               vel_bin    : number of velocity bins shifted, 0 means no velocity
                            can take non-integer values
                            val_bin > 0
            output:
               signal : size [1 x sig_len]
        '''

        target_vel = self.lam * self.dD * vel_bin
        target_range = self.dR * range_bin

        sig_len = len(signal_ref)
        fft_len = 2 ** (next_pow_two(np.ceil(sig_len + np.abs(range_bin))))

        ttt = np.array([x for x in any_seq(1, (fft_len + 1))])
        fff = time_to_freq(ttt)

        signal_received = ifft(ifftshift(fftshift(fft(signal_ref, fft_len, axis=0), axes=0) *
                                         np.exp(-1 * 1j * np.pi * fff * range_bin), axes=0), axis=0)

        signal_received = signal_received * np.exp(-1 * 1j * 2 * np.pi * vel_bin / sig_len * ttt)

        signal_received = signal_received[:sig_len]
        signal_received = signal_received.reshape((1, sig_len))

        return signal_received