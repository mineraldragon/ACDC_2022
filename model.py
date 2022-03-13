import numpy as np
import joblib

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, BatchNormalization
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.advanced_activations import LeakyReLU
#from keras.models import load_model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K
#from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

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
 
        X_train = X_train.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)
        X_test = X_test.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)
        X_validation = X_validation.reshape(-1, Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_validation = np.array(y_validation)

        y_train = np.reshape(y_train,(-1,1))
        y_test = np.reshape(y_test,(-1,1))
        y_validation = np.reshape(y_validation,(-1,1))        

        print(y_train.shape)

        #y_train = to_categorical(y_train)
        #y_test = to_categorical(y_test)
        #y_validation = to_categorical(y_validation)

        num_classes = len(self.calls)

        print('commencing training...')
        input_shape = (Vars.SQUARIFY_SIZE, Vars.SQUARIFY_SIZE, 3)

        print(f"x_train shape: {X_train.shape} - y_train shape: {y_train.shape}")
        print(f"x_test shape: {X_test.shape} - y_test shape: {y_test.shape}")
        #output should be like this:
        #x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
        #x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)


        # Start of Transformer 

        """
        ## Configure the hyperparameters
        """

        learning_rate = 0.001
        weight_decay = 0.0001
        batch_size = 256
        num_epochs = 100
        image_size = 72  # We'll resize input images to this size
        patch_size = 6  # Size of the patches to be extract from the input images
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        transformer_layers = 8
        mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier



        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(X_train)




        """
        ## Implement multilayer perceptron (MLP)
        """

        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x


        """
        ## Implement patch creation as a layer
        """

        class Patches(layers.Layer):
            def __init__(self, patch_size):
                super(Patches, self).__init__()
                self.patch_size = patch_size

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                )
                patch_dims = patches.shape[-1]
                patches = tf.reshape(patches, [batch_size, -1, patch_dims])
                return patches



        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        image = X_train[np.random.choice(range(X_train.shape[0]))]
        plt.imshow(image.astype("uint8"))
        plt.axis("off")

        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size)
        )
        patches = Patches(patch_size)(resized_image)
        print(f"Image size: {image_size} X {image_size}")
        print(f"Patch size: {patch_size} X {patch_size}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")



        """
        ## Implement the patch encoding layer
        The `PatchEncoder` layer will linearly transform a patch by projecting it into a
        vector of size `projection_dim`. In addition, it adds a learnable position
        embedding to the projected vector.
        """


        class PatchEncoder(layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = layers.Dense(units=projection_dim)
                self.position_embedding = layers.Embedding(
                    input_dim=num_patches, output_dim=projection_dim
                )

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded



        """
        ## Build the ViT model
        The ViT model consists of multiple Transformer blocks,
        which use the `layers.MultiHeadAttention` layer as a self-attention mechanism
        applied to the sequence of patches. The Transformer blocks produce a
        `[batch_size, num_patches, projection_dim]` tensor, which is processed via an
        classifier head with softmax to produce the final class probabilities output.
        Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
        which prepends a learnable embedding to the sequence of encoded patches to serve
        as the image representation, all the outputs of the final Transformer block are
        reshaped with `layers.Flatten()` and used as the image
        representation input to the classifier head.
        Note that the `layers.GlobalAveragePooling1D` layer
        could also be used instead to aggregate the outputs of the Transformer block,
        especially when the number of patches and the projection dimensions are large.
        """


        def create_vit_classifier():
            inputs = layers.Input(shape=input_shape)
            # Augment data.
            augmented = data_augmentation(inputs)
            # Create patches.
            patches = Patches(patch_size)(augmented)
            # Encode patches.
            encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

            # Create multiple layers of the Transformer block.
            for _ in range(transformer_layers):
                # Layer normalization 1.
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                # Create a multi-head attention layer.
                attention_output = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=projection_dim, dropout=0.1
                )(x1, x1)
                # Skip connection 1.
                x2 = layers.Add()([attention_output, encoded_patches])
                # Layer normalization 2.
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                # MLP.
                x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
                # Skip connection 2.
                encoded_patches = layers.Add()([x3, x2])

            # Create a [batch_size, projection_dim] tensor.
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.5)(representation)
            # Add MLP.
            features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
            # Classify outputs.
            logits = layers.Dense(num_classes)(features)
            # Create the Keras model.
            model = keras.Model(inputs=inputs, outputs=logits)
            return model



        """
        ## Compile, train, and evaluate the mode
        """


        def run_experiment(model):
            optimizer = tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            )


            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )



            checkpoint_filepath = "/tmp/checkpoint"
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                checkpoint_filepath,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
            )

            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_split=0.1,
                callbacks=[checkpoint_callback],
            )

            model.load_weights(checkpoint_filepath)
            _, accuracy, top_5_accuracy = model.evaluate(X_test, y_test)
            print(f"Test accuracy: {round(accuracy * 100, 2)}%")
            print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

            return history



        vit_classifier = create_vit_classifier()
        history = run_experiment(vit_classifier)

        print('training complete!')



        # End of Transformer  











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
