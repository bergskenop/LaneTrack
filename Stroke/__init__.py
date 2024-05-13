import numpy as np
from helper import *

class Stroke:
    def __init__(self, height_lst):
        self.height_lst = get_filtered_list(interpolate_nan(height_lst))
        self.average_frequency = None
        self.frequency_analysis = np.fft.fft(height_lst)
        self.frequency_analysis[0] = 0 # suppress DC
        self.prevalent_frequency = self.extract_prevalent_freq()
        self.reconstructed_signal = None
        self.stroke_indices = self.local_maximum(height_lst)
        self.stroke_count = len(self.stroke_indices)

    def extract_prevalent_freq(self):
        sorted_peak_indices = np.argsort(np.abs(self.frequency_analysis))[::-1]
        second_most_prevalent_index = sorted_peak_indices[1]
        second_most_prevalent_frequency = np.fft.fftfreq(len(self.height_lst))[second_most_prevalent_index]
        return abs(1/second_most_prevalent_frequency)

    def local_maximum(self, lst, window_size=20):
        local_maxima = []
        for idx, i in enumerate(range(1, len(lst) - 1)):
            if idx > 300:
                if i - window_size >= 0 and i + window_size < len(lst):
                    if lst[i] == max(lst[i - window_size: i + window_size + 1]):
                        local_maxima.append(idx)
        return local_maxima