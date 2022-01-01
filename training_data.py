import os, glob, cv2, copy, random
import numpy as np
import joblib
from random import shuffle
from tqdm import tqdm

from variables import *
from recording import *
from filters import *

class TrainingData:
    # Loads TrainingData instance from disk
    def load():
        home_dir = os.getcwd()
        os.chdir(Vars.MODELS_DIR)
        filename = Vars.TDATA_FILENAME
        print('loading training data from ' + filename + '...')
        tdata = joblib.load(filename)
        print('   complete')
        os.chdir(home_dir)
        return tdata

    def __init__(self):
        self.training_data = {}
        self.testing_data = {}
        self.validation_data = {}

    # Main script to call to prepare both positive and negative data
    def prepare(self):
        home_dir = os.getcwd()
        os.chdir(Vars.TRAINING_DIR)
        calls = list(filter(os.path.isdir, os.listdir(os.getcwd())))

        for call in calls:
            print(call)
            self.training_data[call] = []
            os.chdir(call)
            wavefiles = glob.glob('*.wav')
            shuffle(wavefiles)
            for wavefile in tqdm(wavefiles):
                rec = Recording(wavefile)
                segments = Filters.segmentize_data(rec)
                self.training_data[call].extend(segments)
            # shuffle(self.training_data[call])

            print('split data')
            (self.training_data[call], self.testing_data[call], self.validation_data[call]) = Filters.split_data(self.training_data[call])

            print('augment data')
            self.training_data[call] = Filters.augment_with_synthetic_data(self.training_data[call], Vars.TRAINING_SEGMENTS_PER_CALL)
            self.testing_data[call] = Filters.augment_with_synthetic_data(self.testing_data[call], Vars.TESTING_SEGMENTS_PER_CALL)
            self.validation_data[call] = Filters.augment_with_synthetic_data(self.validation_data[call], Vars.VALIDATION_SEGMENTS_PER_CALL)
            os.chdir('..')
        os.chdir(home_dir)

    # Saves training_data instance
    def save(self, in_models_dir=False):
        home_dir = os.getcwd()
        if in_models_dir:
            os.chdir(Vars.MODELS_DIR)
        filename = Vars.TDATA_FILENAME
        print('saving training data to ' + filename + '...')
        joblib.dump(self, filename)
        print('   complete')
        os.chdir(home_dir)
