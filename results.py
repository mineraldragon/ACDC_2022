import csv, cv2, sys
import joblib
import numpy as np

from variables import *

# Results class saves all the raw results that were stored in a recording object and
# can export CSVs and images of the detected calls to the appropriate results directory.
# 
class Results:
    def load(folder_name):
        home_dir = os.getcwd()
        os.chdir(Vars.RESULTS_DIR)
        os.chdir(folder_name)
        filename = folder_name + '.results'
        results = joblib.load(filename)
        print('loaded results from ' + filename)
        results.__init__(results.name, results.recording)
        os.chdir(home_dir)
        return results

    def __init__(self, name, recording):
        self.name = name
        self.recording = recording

    def export_csv(self, to_results_dir=False):
        print('exporting csv for ' + self.name)
        home_dir = os.getcwd()
        if to_results_dir:
            os.chdir(Vars.RESULTS_DIR)
            os.chdir(self.name)
        labels_file = open(self.recording.file+'.csv', 'w')
        labels_file_writer = csv.writer(labels_file)
        labels_file_writer.writerow(['call','start','end'])
        for label in self.recording.labels:
            labels_file_writer.writerow(label)
        labels_file.close()
        os.chdir(home_dir)
        
        
    def export_Audacity(self, to_results_dir=False):
        print('exporting Audacity labels for ' + self.name)
        home_dir = os.getcwd()
        if to_results_dir:
            os.chdir(Vars.RESULTS_DIR)
            os.chdir(self.name)   
        labels_file = open(self.recording.file+'.txt', 'w', newline='')
        labels_file_writer = csv.writer(labels_file, delimiter='\t')
        q=self.recording.labels
        for label in self.recording.labels:
            q=list(label)
            q[0]=label[1]
            q[1]=label[2]
            q[2]=label[0]
            labels_file_writer.writerow(q)
        labels_file.close()
        os.chdir(home_dir)   
                

    def export_images(self, to_results_dir=False):
        print('exporting images for ' + self.name)
        home_dir = os.getcwd()
        if to_results_dir:
            os.chdir(Vars.RESULTS_DIR)
            os.chdir(self.name)
        for label in self.recording.labels:
            (call, s, e) = label
            spec = np.uint8(self.recording.get_spectrogram_between_timestamps(s, e) * 255)
            filename = str(int(round(s))) + '_' + call + '.png'
            cv2.imwrite(filename, spec)
        os.chdir(home_dir)

    def save(self, to_results_dir=False):
        home_dir = os.getcwd()
        if to_results_dir:
            os.chdir(Vars.RESULTS_DIR)
            os.chdir(self.name)
        filename = self.name + '.results'
        joblib.dump(self, filename)
        print('saved model classifier to ' + filename)
        os.chdir(home_dir)
