import copy
import numpy as np
from tqdm import tqdm

from variables import *
from model import *

# This class takes in a recording and scans through it with all trained models, and then stores
# the results
class Scanner:
    def __init__(self, preload=True, disable_pbar=False):
        self.recording = None
        self.raw_results = {}
        self.smoothed_results = {}
        self.final_results = {}
        self.labels = []
        self.model = None
        self.disable_pbar = disable_pbar
        if preload:
            self.model = Model.load()

    # scan and label a recording with all trained models
    def process(self, recording):
        if not self.disable_pbar:
            print('    scanning, smoothing, and labeling...')
        self.recording = recording
        self.initial_scan()
        self.smoothing()
        self.label()

    # step through recording and grab slices of a given window size corresponding to given model.
    # classify using given model and store raw result value
    def initial_scan(self): #todo add decay
        for call in self.model.calls:
            self.raw_results[call] = []
        self.raw_results['timestamp'] = []

        segment_size = int(round(self.recording.sample_rate * Vars.SEGMENT_LENGTH))
        step_size = int(round(self.recording.sample_rate * Vars.SEGMENT_STEP))
        recording_size = len(self.recording.data)

        s = 0
        e = segment_size
        with tqdm(total=recording_size, unit="recording_secs", unit_scale=True, unit_divisor=self.recording.sample_rate, disable=self.disable_pbar) as pbar:
            while e <= recording_size:
                spec = self.recording.get_spectrogram_between_indices(s, e)
                timestamp = np.mean((s,e)) / float(self.recording.sample_rate)

                model_prediction = self.model.predict_single(spec)
                self.raw_results['timestamp'].append(timestamp)
                for i in range(len(self.model.calls)):
                    call = self.model.calls[i]
                    self.raw_results[call].append(model_prediction[i])

                pbar.update(step_size)
                s += step_size
                e += step_size
        self.recording.raw_results = copy.copy(self.raw_results)

    def smoothing(self, kernel_size=Vars.SMOOTHING_KERNEL_SIZE):
        for call in self.model.calls:
            self.smoothed_results[call] = []
        self.smoothed_results['timestamp'] = copy.copy(self.raw_results['timestamp'])
        half_kernel_size = int(np.floor(kernel_size/2.0))

        calls_minus_noise = copy.copy(self.model.calls)
        calls_minus_noise.remove(Vars.NOISE_STRING)
        for i in tqdm(range(len(self.smoothed_results['timestamp'])), disable=self.disable_pbar):
            s = max(0, i-half_kernel_size)
            e = min(i+half_kernel_size, len(self.smoothed_results['timestamp'])-1)
            for call in calls_minus_noise:
                prev = self.raw_results[call][s:e+1]
                prev = np.mean(prev)
                smoothed = prev * 0.5 + self.raw_results[call][e] * 0.5
                self.smoothed_results[call].append(smoothed)
        self.recording.smoothed_results = copy.copy(self.smoothed_results)

    # add a label for a continuous group of raw results higher than a specified threshold value
    # as long as the length between start and end time is above a proportion of the call length
    def label(self):
        self.final_results['timestamp'] = copy.copy(self.smoothed_results['timestamp'])
        for call in self.model.calls:
            self.final_results[call] = [0] * len(self.final_results['timestamp'])

        calls_minus_noise = copy.copy(self.model.calls)
        calls_minus_noise.remove(Vars.NOISE_STRING)
        for call in tqdm(calls_minus_noise, disable=self.disable_pbar):
            self.detect_call(call)
        self.labels.sort(key=lambda x: x[1])

        self.recording.final_results = copy.copy(self.final_results)
        self.recording.labels = copy.copy(self.labels)

    def detect_call(self, call):
        above = False
        call_start = None
        for i in range(len(self.final_results['timestamp'])):
            prediction = self.smoothed_results[call][i]
            if not above and prediction >= Vars.CONFIDENCE_THRESHOLD:
                above = True
                call_start = self.smoothed_results['timestamp'][i]
                call_start_index = i
            elif above and (prediction < Vars.CONFIDENCE_THRESHOLD or i >= len(self.smoothed_results['timestamp'])-1):
                above = False
                call_end = self.smoothed_results['timestamp'][i]
                call_end_index = i
                detection_length = call_end - call_start
                if detection_length >= (Vars.MIN_DETECTION_LENGTH_RATIO * Vars.WINDOW_LENGTHS[call]):
                    self.labels.append((call, call_start, call_end))
                    self.final_results[call][call_start_index:call_end_index] = [1] * (call_end_index-call_start_index)
