import os, datetime
import numpy as np

from variables import *
from results import *

class Exporter:
    def __init__(self):
        self.recording = None
        self.folder_name = None

    # Takes in a recording file, creates a new folder in the results directory to save all
    # results files. Then creates a Results object and calls on it to save itself and corresponding
    # csv file and images
    def process(self, recording):
        print('exporting results for ' + recording.file)
        self.recording = recording
        self.create_folder()
        results = Results(self.folder_name, self.recording)
        results.export_csv(to_results_dir=True)
        results.export_Audacity(to_results_dir=True)
        
    def create_folder(self):
        home_dir = os.getcwd()
        os.chdir(Vars.RESULTS_DIR)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.folder_name = timestamp + '_' + self.recording.file
        os.makedirs(self.folder_name)
        print('created folder ' + self.folder_name)
        os.chdir(home_dir)
