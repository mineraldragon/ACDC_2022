import os
import numpy as np

class Vars:
    NPERSEG = 256
    NOVERLAP = int(NPERSEG * 0.25)
    WINDOW = 'hanning'
    SPECTROGRAM_RAW_LOW = 1
    SPECTROGRAM_RAW_HIGH = 4
    SPECTROGRAM_POWER_FACTOR = 4
    LOWCUT = 4500
    HIGHCUT = 9500
    SPECTROGRAM_HEIGHT = int(64)
    SQUARIFY_SIZE = 64
    MORPH_CLEAN_KERNEL = np.ones((3,3))
    ROTATIONS = (-2, 2)
    SHEARS_HORIZ = (-2, 2)
    SHEARS_VERT = (-3, 3)
    TILTS_HORIZ = (-8, 8)
    TILTS_VERT = (-8, 8)
    STRETCHES_VERT = (-16, 6)
    ADJUST_BRIGHTNESS = (0.5, 2)
    MINIMUM_VALUE = 0.01
    MINIMUM_AVG_VALUE = 0.001
    MAXIMUM_AVG_VALUE = 0.9
    TRAINING_DIR = 'training_data'
    RECORDINGS_DIR = 'recordings'
    RESULTS_DIR = 'results'
    MODELS_DIR = 'models'
    CONFIDENCE_THRESHOLD = 0.9
    TRAINING_BATCH_SIZE = 128
    TRAINING_EPOCHS = 8
    DETECTION_LENGTH_RATIO = 0.5
    WINDOW_LENGTHS = {'Chi': 0.25,'Tr': 0.25,'Ph': 0.40,'Tw': 0.5}
    SEGMENT_LENGTH = 0.45
    SEGMENT_STEP = 0.04
    VALIDATION_RATIO = 0.2
    TEST_RATIO = 0.1
    TRAINING_SEGMENTS_PER_CALL = 20000
    TESTING_SEGMENTS_PER_CALL = int(round(TRAINING_SEGMENTS_PER_CALL * TEST_RATIO))
    VALIDATION_SEGMENTS_PER_CALL = int(round(TRAINING_SEGMENTS_PER_CALL * VALIDATION_RATIO))
    TDATA_FILENAME = 'acdc.tdata'
    NOISE_STRING = 'Noise'
    MODEL_FILENAME = 'acdc.model'
    MODEL_FILENAME = 'saved_model.pb'
    MODEL_ATTR_FILENAME = 'acdc.modelattr'
    MODEL_CMATRIX_FILENAME = 'acdc_model.png'
    MIN_DETECTION_LENGTH_RATIO = 0.2
    SMOOTHING_KERNEL_SIZE = 5
    VOLUME_AMP_MULTIPLE = 60
