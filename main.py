import os
import sys
import logging
import numpy as np

from simulator import Signal
from utils import make_dir

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

log = logging.getLogger()

def log_setup():
    log.setLevel(logging.DEBUG)
    file = logging.FileHandler('results/log.log', mode = 'w')
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    log.addHandler(file)
    hand_stdout = logging.StreamHandler(sys.stdout)
    hand_stdout.setLevel(logging.DEBUG)
    hand_stdout.setFormatter(formatter)
    log.addHandler(hand_stdout)



def main():
    make_dir('result')
    log_setup()

    signal = Signal()

    signal_ref, received_signal = signal.simulate()
    logging.info('Simulating Signal Complete')

    data_mf, (rv_map, rv_map_log) = signal.process(signal_ref, received_signal)
    logging.info('Processing Signal Complete')

    """signal.plot_rv_map(rv_map_log)
    logging.info('Plotting RV Map Complete')"""

    # i_row, i_col = signal.detect_target(rv_map_log)
    # logging.info('Detecting Targets Complete')

    # signal.estimate_doa(data_mf, i_row, i_col)
    # logging.info('Estimating DOA Complete')


if __name__ == '__main__':
    main()


