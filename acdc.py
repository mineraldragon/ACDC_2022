import glob, os
from variables import *

# Reads .wav files in training_data directory, processes them, and outputs a .tdata file
# with the aggregated training data that can be used by the model for training later on
def prepare_training_data():
    print("==PREPARE TRAINING DATA==")
    from training_data import TrainingData
    training_data = TrainingData()
    training_data.prepare()
    training_data.save(in_models_dir=True)

# Trains Keras/Tensorflow model by loading .tdata file from disk, fitting, saving, and finally
# evaluating the resulting model accuracy
def train():
    print("==TRAIN ALL MODELS==")
    from model import Model
    model = Model()
    model.train()
    model.save(in_models_dir=True)
    Model.evaluate()

def evaluate():
    print("==EVALUATE MODELS==")
    from model import Model
    Model.evaluate()

# Processes recordings in the recordings folder using the saved Keras model and outputs
# call labels in .csv format and .txt format (tab-delimited Audacity readable) into the results directory.
# A new results directory is created each time this is run, as a sub-directory within 'results'. 
# Results directories are named according to the date and time of the run, like this:
# [YYYYMMDD]_[HHMMSS]_[recording filename]
# It will process .wav files in the recordings directory regardless of whether they have been
# processed before, but will store unique results files for each time processed
def process():
    print("==PROCESS RECORDINGS==")
    from scanner import Scanner
    from exporter import Exporter
    from recording import Recording
    home_dir = os.getcwd()
    os.chdir(Vars.RECORDINGS_DIR)
    recordings = []
    wavefiles = glob.glob('*.wav')
    for wavefile in wavefiles:
        recordings.append(Recording(wavefile))
    os.chdir(home_dir)
    print('processing the following recordings: ' + str(wavefiles))
    model = Scanner().model
    for recording in recordings:
        print(' ')
        print('processing ' + recording.file)
        scanner = Scanner(preload=False)
        scanner.model = model
        exporter = Exporter()
        scanner.process(recording)
        exporter.process(recording)
        print(' ')

# List all command line interface options
def shortcuts_all():
    sc = [
        ('prepare training data', prepare_training_data),
        ('train models', train),
        ('process recordings', process),
        ('evaluate models', evaluate),
        ('exit', None)
    ]
    return sc


# Controller function for the command line interface
def controller():
    shortcuts = None
    shortcuts = shortcuts_all()

    while True:
        print(' ')
        print("==ANIMAL CALL DETECTION AND CLASSIFICATION (ACDC)==")
        for i in range(len(shortcuts)):
            print(str(i) + ') ' + shortcuts[i][0])
        selection = input('enter number for the action you would like to perform: ')
        print(' ')
        selection = int(selection)

        if shortcuts[selection][0] == 'exit':
            break

        shortcuts[selection][1]()

if __name__ == "__main__":
    controller() 
