import numpy as np
import joblib

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import tensorflow as tf

from training_data import TrainingData
from variables import *
from filters import *

class Model:
    # This class method loads a model file from disk in two parts
    # 1) The raw savefile for the neural network and weights
    # 2) Additional attributes for the model class (e.g. name, score)
    def load():
        home_dir = os.getcwd()
        os.chdir(Vars.MODELS_DIR)
        model = Model()
        filename = Vars.MODEL_ATTR_FILENAME
        attr = joblib.load(filename)
        print('loaded model attributes from ' + filename)
        model.calls = attr[0]
        model.score = attr[1]
        model.cmatrix = attr[2]
        filename = Vars.MODEL_FILENAME
        model.classifier = load_model(filename)
        print('loaded model classifier from ' + filename)
        os.chdir(home_dir)
        return model

    # Run evaluation script and save confusion matrix to disk
    def evaluate():
        model = Model.load()

        tdata = TrainingData.load()
        (X_train, y_train), (X_test, y_test), (X_validation, y_validation) = model.combine_and_add_targets(tdata)

        X_test = model.prefilter(X_test)
        X_validation = model.prefilter(X_validation)

        y_test = to_categorical(y_test)
        y_validation = to_categorical(y_validation)

        model.score = model.classifier.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', model.score[0])
        print('Test accuracy:', model.score[1])

        X = np.concatenate((X_test, X_validation))
        y = np.concatenate((y_test, y_validation))

        y_pred = model.classifier.predict(X)
        cmatrix = confusion_matrix(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))

        print(model.calls)
        print(cmatrix)
        ax = sns.heatmap(cmatrix, annot=True, fmt='d', xticklabels=model.calls, yticklabels=model.calls)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xlabel('predicted')
        plt.ylabel('actual')
        fig = ax.get_figure()
        fig.savefig(Vars.MODELS_DIR + '/' + Vars.MODEL_CMATRIX_FILENAME)
        plt.show()

    def __init__(self):
        self.classifier = None
        self.calls = None
        self.score = None
        self.cmatrix = None

    # Loads the training data pertaining to the call type, then creates and trains the model
    def train(self):

        print('gathering training data...')
        tdata = TrainingData.load()
        (X_train, y_train), (X_test, y_test), (X_validation, y_validation) = self.combine_and_add_targets(tdata)

        X_train = self.prefilter(X_train)
        X_test = self.prefilter(X_test)
        X_validation = self.prefilter(X_validation)
        #X_train = X_train[:, :, 0, :]
        #X_test = X_test[:, :, 0, :]
        #X_validation = X_validation[:, :, 0, :]

        print(X_train.shape)
        X_train = X_train.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)
        X_test = X_test.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)
        X_validation = X_validation.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_validation = to_categorical(y_validation)

        num_classes = len(self.calls)

        print('commencing training...')
        input_shape = (Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)




        model = Sequential()

        pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                           input_shape=input_shape,
                           pooling='avg',
                           classes=num_classes,
                           weights='imagenet')
        for layer in pretrained_model.layers:
                layer.trainable=False

        model.add(pretrained_model)


        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))






        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])
        print('printed model')

        temp_filepath = Vars.MODELS_DIR+'/'+Vars.MODEL_FILENAME
        
        save_best_model = ModelCheckpoint(filepath=temp_filepath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True)

        model.fit(X_train, y_train,
                  batch_size=Vars.TRAINING_BATCH_SIZE,
                  epochs=Vars.TRAINING_EPOCHS,
                  verbose=1,
                  callbacks=[save_best_model],
                  validation_data=(X_validation, y_validation))

        self.classifier = model
        print('training complete!')

    # Uses the trained classifier to make a prediction on a single input
    def predict_single(self, x):
        if not Filters.simple_check(x):
            return [0]*len(self.calls)
        X = self.prefilter([x])
        result = self.classifier.predict(X)[0]
        return result

    # Prefilter images
    def prefilter(self, X):
        Xp = []
        for i in range(len(X)):
            x = Filters.squarify(X[i])
            x = Filters.gray2rgb(x)
            x = Filters.rescale(x)
            x = np.expand_dims(x, 2)
            Xp.append(x)
        Xp = np.array(Xp)
        return Xp

    # Combine the different call training data and add the targets (i.e. the correct labels)
    def combine_and_add_targets(self, tdata):
        X_train, y_train = self.get_X_y(tdata.training_data)
        X_test, y_test = self.get_X_y(tdata.testing_data)
        X_validation, y_validation = self.get_X_y(tdata.validation_data)
        return (X_train, y_train), (X_test, y_test), (X_validation, y_validation)

    # Helper function for above
    def get_X_y(self, data):
        calls = list(data.keys())
        calls.sort()
        calls.remove(Vars.NOISE_STRING)
        calls.append(Vars.NOISE_STRING)

        if type(self.calls) == type(None):
            self.calls = calls

        X = []
        y = []
        for i in range(len(calls)):
            call_data = data[calls[i]]
            X.extend(call_data)
            y.extend([i]*len(call_data))
        return X, y

    # Save trained model to model directory in the afortmentioned two parts
    def save(self, in_models_dir=False):
        home_dir = os.getcwd()
        if in_models_dir:
            os.chdir(Vars.MODELS_DIR)
        print('saving model...')
        filename = Vars.MODEL_FILENAME
        self.classifier.save(filename)
        attr = [self.calls, self.score, self.cmatrix]
        filename = Vars.MODEL_ATTR_FILENAME
        joblib.dump(attr, filename)
        print('   complete')
        os.chdir(home_dir)
