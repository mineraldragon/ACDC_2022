import glob, os
from variables import *

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
    for recording in recordings:
        print('processing ' + recording.file)
        scanner = Scanner()
        exporter = Exporter()
        scanner.process(recording)
        exporter.process(recording)
        print(' ')

process()
