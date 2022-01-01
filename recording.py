import cv2
import numpy as np
# from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile

from variables import *
from filters import *

# In order to load very long .wav files (e.g. recordings) into memory, this Recording class
# was created. It loads the raw data into memory, but only loads spectrograms between timestamps
# or indices when called upon to do so
class Recording:
    def __init__(self, filename=None):
        self.sample_rate = None
        self.file = None
        self.data = None
        self.raw_results = None
        self.smoothed_results = None
        self.final_results = None
        self.labels = None
        self.length = None

        if type(filename) != type(None):
            self.read(filename)

    def read(self, filename):
        data = wavfile.read(filename)
        self.file = filename[0:-4]
        self.sample_rate = data[0]
        if len(np.shape(data[1])) == 1:
            self.data = data[1]
        else:
            self.data = data[1][:,0]
        self.length = len(self.data) / float(self.sample_rate)

    def get_spectrogram_between_indices(self, s, e):
        data = self.data[s:e+1]
        spec = Filters.create_spectrogram(data, self.sample_rate)
        return spec

    def get_spectrogram_between_timestamps(self, s, e):
        s = self.timestamp_to_data_index(s)
        e = self.timestamp_to_data_index(e)
        data = self.data[s:e+1]
        spec = Filters.create_spectrogram(data, self.sample_rate)
        return spec

    def timestamp_to_data_index(self, timestamp):
        return min(int(round(self.sample_rate * timestamp)), len(self.data)-1)
