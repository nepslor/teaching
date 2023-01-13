import numpy as np


def merge_ramps(x, x_high=0.5, x_low=0.1, x_filt_1=None, x_filt_2=None):
    """
    Merge first order derivative such as jumps occurring in between two sampling times are LIKELY joined together.
    :param x: np.ndarray of the signal
    :param x_high: threshold for identifying interesting derivatives
    :param x_low: threshold for merging derivatives
    :return:
    """
    x_diff = np.diff(x.ravel())
    if x_filt_1 is None:
        x_filt_1 = np.where(x_diff > x_high)[0]
    for f in x_filt_1:
        if f >= len(x_diff) - 1:
            continue
        x_f = x_diff[f - 1:f + 2]
        # if np.any(x_f[[0, 2]] > x_low):
        x_diff[f + 1] = np.sum(x_f[x_f > x_low])
        x_diff[f] = 0

    if x_filt_2 is None:
        x_filt_2 = np.where(x_diff < -x_high)[0]
    for f in x_filt_2:
        if f >= len(x_diff) - 1:
            continue
        x_f = x_diff[f - 1:f + 2]
        # if np.any(x_f[[0, 2]] < -x_low):
        x_diff[f] = np.sum(x_f[x_f < -x_low])
        x_diff[f + 1] = 0

    # x_diff[np.abs(x_diff) <= x_high] = 0 #should be OK to do this!
    return x_diff, x_filt_1, x_filt_2