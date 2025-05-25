import sys

sys.path.append('../')
import numpy as np
import datetime
from typing import List
from functools import wraps
from time import time
from downloading.utils import calculate_and_save_best_images


def timing(f):
    """ Decorator used to time function execution times
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print(f'{f.__name__}, {np.around(te-ts, 2)}')
        return result

    return wrap


def extract_dates(date_dict: dict, year: int) -> List:
    """ Transforms a SentinelHub date dictionary to a
         list of integer calendar dates indicating the date of the year
    """
    dates = []
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    starting_days = np.cumsum(days_per_month)
    for date in date_dict:
        dates.append(((date.year - year) * 365) +
                     starting_days[(date.month - 1)] + date.day)
    return dates


def to_int16(array: np.array) -> np.array:
    '''Converts a float32 array to uint16'''
    assert np.min(array) >= 0, np.min(array)
    assert np.max(array) <= 1, np.max(array)

    array = np.clip(array, 0, 1)
    array = np.trunc(array * 65535)
    assert np.min(array >= 0)
    assert np.max(array <= 65535)

    return array.astype(np.uint16)


def to_float32(array: np.array) -> np.array:
    """Converts an int array to float32"""
    #print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.divide(np.float32(array), 65535.)
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    return array


def process_sentinel_1_tile(sentinel1: np.ndarray,
                            dates: np.ndarray) -> np.ndarray:
    """Converts a (?, X, Y, 2) Sentinel 1 array to a regular monthly grid

        Parameters:
         sentinel1 (np.array):
         dates (np.array):

        Returns:
         s1 (np.array)
    """
    s1, _ = calculate_and_save_best_images(sentinel1, dates)
    monthly = np.zeros((12, sentinel1.shape[1], sentinel1.shape[2], 2), dtype = np.float32)
    index = 0
    for start, end in zip(
            range(0, 24 + 2, 24 // 12),  #0, 72, 6
            range(24 // 12, 24 + 2, 24 // 12)):  # 6, 72, 6
        monthly[index] = np.median(s1[start:end], axis=0)
        index += 1
    #print(np.mean(monthly, axis = (1, 2, 3)))
    return monthly



def _check_for_alt_img(probs, dates, date):
    # Checks to see if there is an image within win days that has
    # Less local cloud cover. If so, remove the higher CC image.
    # This is done to avoid having to remove (interpolate)
    # All of an image for a given tile, as this can cause artifacts
    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341, 410]

    begins = (end - date)
    begins[begins < 0] = 999
    month_start = begin[np.argmin(begins)]
    month_end = end[np.argmin(begins)]
    lower = month_start
    upper = month_end

    candidate_idx = np.argwhere(
        np.logical_and(np.logical_and(dates >= lower, dates <= upper),
                       dates != date))
    candidate_probs = probs[candidate_idx]
    if len(candidate_probs) == 0:
        return False
    if np.min(candidate_probs) < 0.3:
        return True
    else:
        return False




def remove_noise_clouds(arr):
    # global cloudmasks from S2cloudless or SCL are noisy. They can have
    # persistent commission errors, which this fn helps mitigate.
    for t in range(arr.shape[0]):
        for x in range(1, arr.shape[1] - 1, 1):
            for y in range(1, arr.shape[2] - 1, 1):
                window = arr[t, x - 1:x + 2, y - 1:y + 2]
                if window[1, 1] > 0:
                    # if one pixel and at least (n - 2) are cloudy
                    if np.sum(window > 0) <= 1 and np.sum(
                            arr[:, x, y]) > arr.shape[0] - 1:
                        window = 0.
                        arr[t, x - 1:x + 2, y - 1:y + 2] = window
    return arr


