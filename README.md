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

<code>variables.py</code> contains constants used in various modules. Some of them are highlighted here because changing their values according to the use's needs can help get better results.  

<code>CONFIDENCE_THRESHOLD</code>
This is the value that needs to be exceeded in the the final layer of the model to trigger detection of a call. Lowering this value makes the model more likely to detect something but can lead to more false postives. Raising this value makes the model less likely to detect something but reduces false postitives.

<code>TRAINING_EPOCHS</code>
This determines the number of iterations that the model trains. We have good experience using at least 10 epochs. 

<code>WINDOW_LENGTHS = {'Chi': 0.25,'Tr': 0.25,'Ph': 0.40,'Tw': 0.5}</code>
Window lengths in seconds are set for each vocalization type. The names of the calls ‘Chi’, ‘Tr’ ‘Ph’ and ‘Tw’ have to correspond to folder names in the ‘training_data’ folder. If different or additional classes need to be trained, this variable needs to change accordingly

<code>TRAINING_SEGMENTS_PER_CALL</code>
This is a target number of segments which determines whether the data needs to be augmented. It makes sense to set this value equal to the class with the most segments so that other classes are augmented and get the same number, removing class imbalance. 

<code>VOLUME_AMP_MULTIPLE</code>
This variable determines by how much the data should be amplified. There is a threshold being applied so segments that do not cross the threshold are discarded. Change this value to get the optimal balance between false positives and false negatives. 











Collaborators
Samvaran Sharma, Karthik Srinivasan, and Rogier Landman

This project was developed in collaboration with MIT Brain and Cognitive Sciences (c) 2016-2022.
