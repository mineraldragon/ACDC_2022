import cv2, copy, random
import numpy as np
from scipy.signal import spectrogram
from tqdm import tqdm

from variables import *

# Helper functions and filters that are used commonly in the rest of the codebase
class Filters:
    # Given raw data and sample rate, creates a spectrogram image and time series information
    # for spectrogram columns
    def create_spectrogram(data, sample_rate):
        data = np.float32(data) * Vars.VOLUME_AMP_MULTIPLE
        data = np.int16(np.clip(data, -32768, 32767))
        if len(data) / float(sample_rate) > 60:
            return None
        f, t, spec = spectrogram(data,
                                 fs=float(sample_rate),
                                 window=Vars.WINDOW,
                                 nperseg=Vars.NPERSEG,
                                 noverlap=Vars.NOVERLAP)
        lowcut_index = np.searchsorted(f, Vars.LOWCUT)
        highcut_index = np.searchsorted(f, Vars.HIGHCUT)
        spec = spec[lowcut_index:highcut_index, :]

        if np.min(spec) == 0.0:
            spec[spec == 0.0] = 0.0001
        spec = np.log10(spec)

        spec = np.clip(spec, Vars.SPECTROGRAM_RAW_LOW, Vars.SPECTROGRAM_RAW_HIGH)
        spec = np.add(spec, -1*Vars.SPECTROGRAM_RAW_LOW)
        spec = np.power(spec, Vars.SPECTROGRAM_POWER_FACTOR)
        spec = np.divide(spec, (Vars.SPECTROGRAM_RAW_HIGH-Vars.SPECTROGRAM_RAW_LOW)**Vars.SPECTROGRAM_POWER_FACTOR)
        spec = np.flipud(spec)
        spec = cv2.resize(spec, (len(spec[0]), Vars.SPECTROGRAM_HEIGHT))
        return spec

    # Splits data from recording object into overlapping spectrogram segments
    def segmentize_data(rec):
        segment_size = int(round(rec.sample_rate * Vars.SEGMENT_LENGTH))
        step_size = int(round(rec.sample_rate * Vars.SEGMENT_STEP))

        data = np.pad(rec.data, pad_width=(segment_size-step_size), mode='constant', constant_values=0)

        segments = []
        s = 0
        e = segment_size
        while e <= len(data):
            segment_data = data[s:e]
            spec = Filters.create_spectrogram(segment_data, rec.sample_rate)
            if Filters.simple_check(spec):
                segments.append(spec)
            s += step_size
            e += step_size

        return segments

    # Resizes input spectrogam to be a square
    def squarify(x):
        return cv2.resize(x, (Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE))

    def gray2rgb(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    # Rescales spectrogram image data to be from 0->1 to -1->1 to better fit the neural network
    def rescale(x):
        return (x*2)-1

    # Adds border around spectrogram for cleaner visualization
    def border(x):
        x = np.pad(x, pad_width=1, mode='constant', constant_values=1)
        return x

    # Morphological operation to clean spectrogram image
    # def morph_clean(x):
    #     x = cv2.morphologyEx(x, cv2.MORPH_OPEN, Vars.MORPH_CLEAN_KERNEL)
    #     return x

    # Center spectrogram image by centroid
    # def center(x):
    #     if np.sum(x) == 0:
    #         return x
    #     M = cv2.moments(x)
    #     cx = int(M['m10']/M['m00'])
    #     cy = int(M['m01']/M['m00'])
    #     (height, width) = np.shape(x)
    #     shiftx = round(width/2.0) - cx
    #     shifty = round(height/2.0) - cy
    #     t = np.float32([[1,0,shiftx],[0,1,shifty]])
    #     return cv2.warpAffine(x, t, (width, height))

    # Rotate input spectrogram image by theta degrees
    def rotate(x, theta):
        rows, cols = np.shape(x)
        midrow = round(rows/2.0)
        midcol = round(cols/2.0)
        M = cv2.getRotationMatrix2D((midcol,midrow), -1*theta, 1)
        return cv2.warpAffine(x, M, (cols, rows))

    # Shear input spectrogram image by given pixels in each direction
    def shear(x, horiz=0, vert=0):
        rows, cols = np.shape(x)
        pts1 = np.float32([[round(cols*0.33), round(rows*0.67)],
                           [round(cols*0.67), round(rows*0.67)],
                           [round(cols*0.67), round(rows*0.33)]])
        pts2 = np.float32([[round(cols*0.33), round(rows*0.67)+vert],
                           [round(cols*0.67), round(rows*0.67)],
                           [round(cols*0.67)+horiz, round(rows*0.33)]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(x, M, (cols, rows))

    # Stretch input spectrogram by given pixels in each direction
    def stretch(x, horiz=0, vert=0):
        rows, cols = np.shape(x)
        vert_up = int(np.ceil(vert/2.0))
        vert_down = int(np.floor(vert/2.0))
        horiz_left = int(np.ceil(horiz/2.0))
        horiz_right = int(np.floor(horiz/2.0))
        pts1 = np.float32([[round(cols*0.33), round(rows*0.67)],
                           [round(cols*0.67), round(rows*0.67)],
                           [round(cols*0.67), round(rows*0.33)]])
        pts2 = np.float32([[round(cols*0.33)-horiz_left, round(rows*0.67)+vert_down],
                           [round(cols*0.67)+horiz_right, round(rows*0.67)+vert_down],
                           [round(cols*0.67)+horiz_right, round(rows*0.33)-vert_up]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(x, M, (cols, rows))

    # Tilt image horizontally
    def tilt(x, horiz=0, vert=0):
        rows, cols = np.shape(x)
        pts1 = np.float32([[0, 0],
                           [cols-1, 0],
                           [0, rows-1],
                           [cols-1, rows-1]])
        pts2 = copy.copy(pts1)
        if horiz > 0:
            pts2[1,1] = pts2[1,1] + horiz
            pts2[3,1] = pts2[3,1] - horiz
        if horiz < 0:
            horiz = -1*horiz
            pts2[0,1] = pts2[0,1] + horiz
            pts2[2,1] = pts2[2,1] - horiz
        if vert > 0:
            pts2[2,0] = pts2[2,0] + vert
            pts2[3,0] = pts2[3,0] - vert
        if vert < 0:
            vert = -1*vert
            pts2[0,0] = pts2[0,0] + vert
            pts2[1,0] = pts2[1,0] - vert
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(x, M, (cols, rows))

    # Adjust image brightness
    def adjust_brightness(x, delta):
        return np.clip(x * delta, 0.0, 1.0)

    # Simple check to determine if there is anything of value in an image, or if it mostly
    # blank or mostly noise
    def simple_check(x):
        if np.max(x) < Vars.MINIMUM_VALUE:
            return False
        if np.mean(x) < Vars.MINIMUM_AVG_VALUE:
            return False
        if np.mean(x) > Vars.MAXIMUM_AVG_VALUE:
            return False
        return True

    # Train-Test-Validation split
    # Assumes files have been shuffled and then segmented
    # hence, segments in order, but sections of segments corresponding to files randomized
    def split_data(data):
        num_validation = int(np.ceil(len(data) * Vars.VALIDATION_RATIO))
        num_test = int(np.ceil(len(data) * Vars.TEST_RATIO))

        validation = data[0:num_validation]
        test = data[num_validation:(num_validation + num_test)]
        train = data[(num_validation + num_test):]

        return (train, test, validation)

    # Given an input image, apply random image transforms to spectrogram
    def create_synthetic_segment(segment):
        rotation = random.randint(Vars.ROTATIONS[0], Vars.ROTATIONS[1])
        shear = (random.randint(Vars.SHEARS_HORIZ[0], Vars.SHEARS_HORIZ[1]), random.randint(Vars.SHEARS_VERT[0], Vars.SHEARS_VERT[1]))
        tilt = (random.randint(Vars.TILTS_HORIZ[0], Vars.TILTS_HORIZ[1]), random.randint(Vars.TILTS_VERT[0], Vars.TILTS_VERT[1]))
        stretch = (0, random.randint(Vars.STRETCHES_VERT[0], Vars.STRETCHES_VERT[1]))
        adjust_brightness = random.uniform(Vars.ADJUST_BRIGHTNESS[0], Vars.ADJUST_BRIGHTNESS[1])
        segment = Filters.rotate(segment, rotation)
        segment = Filters.shear(segment, horiz=shear[0], vert=shear[1])
        segment = Filters.tilt(segment, horiz=tilt[0], vert=tilt[1])
        segment = Filters.stretch(segment, horiz=stretch[0], vert=stretch[1])
        segment = Filters.adjust_brightness(segment, adjust_brightness)
        return segment

    # Runs create_synthetic_segment as many times as necessary to
    def augment_with_synthetic_data(data, target_number):
        synthetic_segments = []
        num_original_segments = len(data)
        num_segments_to_fill = target_number - num_original_segments

        print('   augmenting ' + str(num_original_segments) + ' segments to ' + str(target_number) + ' segments')

        if num_original_segments == target_number:
            return data
        elif num_original_segments > target_number:
            return data[0:target_number]

        for i in tqdm(range(num_segments_to_fill)):
            segment = data[i%num_original_segments]
            segment = Filters.create_synthetic_segment(segment)
            synthetic_segments.append(segment)

        data.extend(synthetic_segments)
        return data
