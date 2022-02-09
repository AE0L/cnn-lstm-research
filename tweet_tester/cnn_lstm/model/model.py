import os
import sys
from io import StringIO
from celery.local import class_property
from celery_progress.backend import ProgressRecorder
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import reshape
from keras.layers.normalization.batch_normalization import BatchNormalization
from twitter.config.model_parameters import setup_params
from keras import models as kmodels
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import Embedding 
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import classification_report, multilabel_confusion_matrix, plot_confusion_matrix
from utilities.logging.log import log

from .metrics import f1_m, precision_m, recall_m


class CNNLSTMModel:
    LABELS = ['Anxiety', 'Depression', 'None']

    def __init__(self, tokenizer=None, embedding_matrix=None, params=None):
        log('Initializing the model')
        module_dir = os.path.dirname(__file__)
        self.model_file_path = os.path.join(module_dir, 'res/cnn-lstm-model')

        # CHECK IF THERE IS A MODEL SAVED
        if os.path.exists(self.model_file_path):
            print('LOADING MODEL...')
            self.model = kmodels.load_model(
                self.model_file_path,
                custom_objects={
                    'f1_m': f1_m,
                    'precision_m': precision_m,
                    'recall_m': recall_m
                }
            )
        else:
            # Embedding Layer
            input_dim = len(tokenizer.word_index) + 1
            output_dim = params['embedding']['OUTPUT_DIM']
            input_length = params['embedding']['INPUT_LENGTH']
            weights = [embedding_matrix]
            # Dense
            dense_units = params['dense']['UNITS']
            dense_act = params['dense']['ACTIVATION']
            # Compile
            metrics = ['accuracy', f1_m, precision_m, recall_m]

            embedding_layer = Embedding(
                input_dim,
                output_dim,
                input_length=input_length,
                weights=weights,
                trainable=False
            )

            # CREATE THE MODEL
            model = Sequential()
            model.add(embedding_layer)

            model.add(Conv1D(64, 3, padding='same', activation='relu'))
            model.add(MaxPooling1D(2))

            model.add(Dropout(0.2))

            model.add(BatchNormalization())

            model.add(LSTM(100))

            model.add(Dense(3, activation=dense_act))

            adam = Adam(learning_rate=0.01)
            rms = RMSprop(learning_rate=0.1, decay=0.0)

            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=metrics)

            self.model = model
            self.model.save(self.model_file_path)
            print(model.summary())

            log('Model successfully initialized')

    def train(self, x_train, y_train, x_val, y_val, epochs=10):
        verbose = 2
        log('Training the model')
        log(f'Train dataset size: {len(x_train)}')
        log(f'Test dataset size: {len(x_val)}')

        class_weight = {
            0: 1.0,
            1: 1.0,
            2: 17.0
        }

        # TRAIN MODEL
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            verbose=verbose,
            class_weight=class_weight,
            batch_size=64
        )

        log('Model training completed')
        log('Evaluating Model')

        # TEST MODEL
        y_test = np.array(y_val)
        y_scores = np.array(self.model.predict(x_val))
        threshold = 0.8
        y_pred = []
        for sample in y_scores:
            y_pred.append([1 if i > threshold else 0 for i in sample])
        y_pred = np.array(y_pred)
        report = classification_report(
            y_test,
            y_pred,
            # np.argmax(y_pred, axis=1),
            target_names=self.LABELS
        )

        # GET MODEL'S STATISTICS
        loss, acc, f1_score, precision, recall = self.model.evaluate(
            x_val, y_val)

        log('Saving model')

        # SAVE MODEL
        self.model.save(self.model_file_path)

        log('Model successfully saved')

        # PLOT MODEL'S ACCURACY
        acc_fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Train and Test accuracy Over Epochs')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        imgdata = StringIO()
        acc_fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        acc_graph = imgdata.getvalue()
        plt.close()

        loss_fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Train and Test loss Over Epochs')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        imgdata = StringIO()
        loss_fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        loss_graph = imgdata.getvalue()
        plt.close()

        # PLOT CONFUSION MATRIX
        cm_fig, ax = plt.subplots(1, 3, figsize=(10, 2))
        cm = multilabel_confusion_matrix(y_test, y_pred)

        for axes, cfs_matrix, label in zip(ax.flatten(), cm, ['Anxiety', 'Depression', 'None']):#, 'None']):
            print_confusion_matrix(cfs_matrix, axes, label, ['N', 'Y'])

        cm_fig.tight_layout()
        plt.close()

        cm_fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        cm_graph = imgdata.getvalue()

        return {
            'acc_graph': acc_graph,
            'loss_graph': loss_graph,
            'cm_graph': cm_graph,
            'report': report,
            'stats': {
                'loss': '{:.2f}'.format(loss),
                'accuracy': '{:.2f}'.format(acc),
                'f1_score': '{:.2f}'.format(f1_score),
                'precision': '{:.2f}'.format(precision),
                'recall': '{:.2f}'.format(recall)
            }
        }

    def test(self, test_seq):
        log('Classifying tweets')

        preds = self.model.predict(test_seq)

        log('Getting average prediction')
        pred_avg = np.average(preds, axis=0)
        log('Tweets classification completed')

        return {'user_pred': pred_avg, 'preds': preds.tolist()}


class PRCallback(Callback):
    def __init__(self, task, max_len):
        self.pr = ProgressRecorder(task)
        self.max_len = max_len

    def on_test_batch_end(self, batch, logs=None):
        self.pr.set_progress(
            batch,
            self.max_len,
            f'Predicting class of {batch} tweet out of {self.max_len} tweets'
        )


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=12):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True Class')
    axes.set_xlabel('Predicted Class')
    axes.set_title(class_label)
