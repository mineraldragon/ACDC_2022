# ACDC_2022
ACDC Version 2022 is a Python 3.7 release of the Animal Call Detection and Classification code powered by Keras/Tensorflow.

## Main Goal
While studying the social behavior of animals in a laboratory setting, a wealth of unstructured data is typically generated. These data, such as hours-long video and/or audio recordings, are tedious for humans to analyze manually. ACDC is a project that seeks to solve this problem for audio analysis - helping researchers to train a neural network and then automatically detect and classify animal calls, outputting the results in a convenient format.

## Overall Flow
To train, the user places training data for discrete call types in the <code>training_data</code> folder and then instructs ACDC to develop detection and classification models based on that data.
To process recordings, the user places recordings in the <code>recordings</code> folder and tells ACDC to process them, resulting in files containing call labels being generated and placed in an output folder. 
Operation is mainly through a command line interface with numbered options which allows the user to enter a number for the action to perform. 

## Most recent changes
- Removal of lesser used features that produced errors
- Fixed Joblib issue
- Use newer version of Keras
- Save to Audacity labels format
- Model save and load debugged
- Installation usign Reguirements.txt

## Approach for detection
The full recording is split up into overlapping segments, each a certain length (e.g. 0.5s). Each segment is fed to the multi-class classifier which determines which type of call that segment contains. Since there is a high degree of overlap between the segments, each section of the spectrogram is essentially covered many times. These results are put in a time series, and the "Scanner" class then goes through the raw results, smoothing them, and then finally discarding continuous segments that are less than a certain proportion of the average call length (e.g. if the average phee call is 1s, and a continuous set of segments were labeled phee, but that contiguous set only lasted 0.3s total, it would be discarded). These steps effectively create a "voting" scheme. If there is a false positive in one segment and one segment only, these steps will likely smooth over them or weed them out. Conversely, if there is a false negative in a sea of true positives, it will not disrupt the chain.

## Installation
1. Download the repo and unzip the files in the directory where you want them
2. Install Anaconda <url>https://www.anaconda.com/</url>
3. Create a new environment with Python 3.7 in Anaconda Navigator
4. Click the environment, click Open Terminal (a command line terminal will open)
5. In the terminal, navigate to the directory that has the ACDC files (where <code>acdc.py</code> is)
6. Type <code>pip install -r requirements.txt</code> and hit enter. Pip should now be installing all the required packages. 

To run ACDC, type <code>python acdc.py</code>. The following menu should appear

<img src="https://github.com/mineraldragon/ACDC_2022/blob/main/img/Menu_screenshot.png" width=50% height=50%>

Now enter the number for the action you want to perform.


## Repo contents
<code>acdc.py</code>This is the main file which calls all other modules.
<code>exporter.py</code>Creates the folder for the results.
<code>filters.py</code>Pre-processing such as creating spectrograms and data augmentation.
<code>model.py</code>Creates, trains, evaluates and saves the model
<code>process.py</code>Manages processing of wave files for detection
<code>recording.py</code>Manages loading of long wave files
<code>results.py</code>Saves the results
<code>scanner.py</code>Steps through a recording and labels the calls
<code>training_data.py</code>Prepares data for training the model
<code>variables.py</code>Constants used in various modules. Edit this to change how the package functions. 

Constants affecting spectrograms and augmentation
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
NOISE_NEG_SAMPLES_RATIO = 0.4
ROTATIONS = (-2, 2)
SHEARS_HORIZ = (-2, 2)
SHEARS_VERT = (-3, 3)
TILTS_HORIZ = (-8, 8)
TILTS_VERT = (-8, 8)
STRETCHES_VERT = (-16, 6)
ADJUST_BRIGHTNESS = (0.5, 2)
MAX_SAMPLES = 150
MINIMUM_VALUE = 0.01
MINIMUM_AVG_VALUE = 0.001
MAXIMUM_AVG_VALUE = 0.9
STEP_LENGTH_RATIO = 0.5


Names of directories
TRAINING_DIR = 'training_data'
RECORDINGS_DIR = 'recordings'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

Constants affecting the model
CONFIDENCE_THRESHOLD = 0.95
(this is the value the final layer in the model needs to exceed to trigger detection of a call)

TRAINING_BATCH_SIZE = 16
TRAINING_EPOCHS = 2
(these determine the number of iterations the model needs to train)

DETECTION_LENGTH_RATIO = 0.5
WINDOW_LENGTHS = {'Chi': 0.25,'Tr': 0.25,'Ph': 0.40,'Tw': 0.5}
(Window lengths in seconds are set for each vocalization type because different vocalizations have different durations. The names of the calls ‘Chi’, ‘Tr’ ‘Ph’ and ‘Tw’ correspond to folder names in the ‘training_data’ folder. If different or additional classes need to be trained, the names in this variable need to added or changes accordingly)

SEGMENT_LENGTH = 0.45
SEGMENT_STEP = 0.04
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1
TRAINING_SEGMENTS_PER_CALL = 100
(this is a target number of segments which determines how many synthetic segments need to be created by the augmentation procedure. It makes sense to make this value equal to the class with the highest number of segments, so that other classes are augmented and get the same number)

TESTING_SEGMENTS_PER_CALL = int(round(TRAINING_SEGMENTS_PER_CALL * TEST_RATIO))
VALIDATION_SEGMENTS_PER_CALL = int(round(TRAINING_SEGMENTS_PER_CALL * VALIDATION_RATIO))

Various filenames
TDATA_FILENAME = 'acdc.tdata'
NOISE_STRING = 'Noise'
MODEL_FILENAME = 'acdc.model'
MODEL_FILENAME = 'saved_model.pb'
MODEL_ATTR_FILENAME = 'acdc.modelattr'
MODEL_CMATRIX_FILENAME = 'acdc_model.png'

MIN_DETECTION_LENGTH_RATIO = 0.2
SMOOTHING_KERNEL_SIZE = 5
VOLUME_AMP_MULTIPLE = 60
(This variable determines by how much the data should be amplified so that enough samples cross the threshold of inclusion in the analysis and not too many background noise samples are erroneously included)











Collaborators
Samvaran Sharma, Karthik Srinivasan, and Rogier Landman

This project was developed in collaboration with MIT Brain and Cognitive Sciences (c) 2016-2022.
