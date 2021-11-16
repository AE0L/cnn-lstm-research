import os
import sys
from io import StringIO
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from twitter.config.model_parameters import setup_params
from keras import models as kmodels
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
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
            # MODEL'S LAYER PARAMETERS
            # Embedding Layer
            input_dim = len(tokenizer.word_index) + 1
            output_dim = params['embedding']['OUTPUT_DIM']
            input_length = params['embedding']['INPUT_LENGTH']
            weights = [embedding_matrix]
            trainable = params['embedding']['TRAINABLE']
            # CNN
            filters = params['cnn']['FILTERS']
            kernel = params['cnn']['KERNEL']
            cnn_act = params['cnn']['ACTIVATION']
            # Max Pooling
            pool_size = params['max-pooling']['POOL_SIZE']
            # strides = params['max-pooling']['STRIDES']
            # Dropout
            dropout = params['dropout']['RATE']
            # LSTM
            lstm_units = params['lstm']['UNITS']
            # Dense
            dense_units = params['dense']['UNITS']
            dense_act = params['dense']['ACTIVATION']
            # Compile
            loss = params['compile']['LOSS']
            optimizer = params['compile']['OPTIMIZER']
            metrics = ['accuracy', f1_m, precision_m, recall_m]

            embedding_layer = Embedding(
                input_dim,
                output_dim,
                input_length=input_length,
                weights=weights,
                trainable=trainable,
            )

            # CREATE THE MODEL
            model = Sequential()
            model.add(embedding_layer)
            model.add(Conv1D(filters, kernel, activation=cnn_act))
            model.add(MaxPooling1D(pool_size))
            model.add(Dropout(dropout))
            model.add(LSTM(lstm_units))
            model.add(Dense(dense_units, activation=dense_act))
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            self.model = model
            self.model.save(self.model_file_path)
            print(model.summary())

            log('Model successfully initialized')

    def train(self, x_train, y_train, x_val, y_val, epochs=10):
        verbose = 2
        log('Training the model')

        # TRAIN MODEL
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            verbose=verbose
        )

        log('Model training completed')
        log('Evaluating Model')

        # TEST MODEL
        y_test = np.argmax(y_val, axis=1)
        y_pred = self.model.predict(x_val)
        report = classification_report(
            y_test,
            np.argmax(y_pred, axis=1),
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
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        imgdata = StringIO()
        acc_fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        acc_graph = imgdata.getvalue()

        # PLOT CONFUSION MATRIX
        cm_fig = plt.figure()
        cm = confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
        cm_df = pd.DataFrame(
            cm,
            index=['Anxiety', 'Depression', 'None'],
            columns=['Anxiety', 'Depression', 'None']
        )

        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        imgdata = StringIO()
        cm_fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        cm_graph = imgdata.getvalue()


        return {
            'acc_graph': acc_graph,
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
