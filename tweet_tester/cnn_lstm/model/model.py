from argparse import ArgumentParser
from cmath import sqrt
import os
import sys
from io import StringIO
from turtle import back
from celery.local import class_property
from celery_progress.backend import ProgressRecorder
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow import reshape
from keras.layers.normalization.batch_normalization import BatchNormalization
from twitter.config.model_parameters import setup_params
from keras import models as kmodels
from keras import regularizers, backend
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Embedding
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import classification_report, multilabel_confusion_matrix, plot_confusion_matrix, precision_recall_curve, roc_curve, f1_score
from utilities.logging.log import log

from .metrics import f1_m, precision_m, recall_m


class CNNLSTMModel:
    CLASS_WEIGHTS = {
        0: 1.0,
        1: 1.0,
        2: 17.0
    }

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
            self.model = self.create_model(tokenizer, embedding_matrix, params)
            self.model.load_weights(os.path.join(module_dir, 'res/models/model_2.h5'))
            self.model.save(self.model_file_path)
            print(self.model.summary())
            log('Model successfully initialized')

        

    def create_model(self, tokenizer, embedding_matrix, params):
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

        # model.add(Dense(3, activation=dense_act))
        model.add(Dense(3, activation=dense_act))

        adam = Adam(learning_rate=0.01)
        rms = RMSprop(learning_rate=0.1, decay=0.0)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=metrics)

        return model


    def model_train(self, model, x_train, y_train, x_val, y_val):
        return model.fit(
        )

    def get_model_name(self, k):
        return 'model_'+str(k)+'.h5'

    def train(self, x_train, y_train, x_val, y_val, epochs=10):
    # def train(self, x, y, epochs=10, tokenizer=None, embedding_matrix=None, params=None):
        # verbose = 2
        # log('Training the model')
        # log(f'Train dataset size: {len(x_train)}')
        # log(f'Test dataset size: {len(x_val)}')

        # skf = KFold(n_splits=5, shuffle=True)
        # val_acc = []
        # val_loss = []
        # val_f1 = []
        # val_precision = []
        # val_recall = []

        # module_dir = os.path.dirname(__file__)
        # save_dir = os.path.join(module_dir, 'res/models')
        # fold_var = 1

        # for index, (train_indices, val_indices) in enumerate(skf.split(x, y)):
        #     print('Training on fold %i/5' % (index + 1))
        #     x_train, x_val = x[train_indices], x[val_indices]
        #     y_train, y_val = y[train_indices], y[val_indices]

        #     model = None
        #     model = self.create_model(tokenizer, embedding_matrix, params)

        #     checkpoint = ModelCheckpoint(
        #         save_dir + '/' + self.get_model_name(fold_var),
        #         monitor='val_accuracy',
        #         verbose=1,
        #         save_best_only=True,
        #         mode='max'
        #     )

        #     callbacks_list = [checkpoint]

        #     history = model.fit(
        #         x_train,
        #         y_train,
        #         validation_data=(x_val, y_val),
        #         verbose=2,
        #         epochs=epochs,
        #         class_weight=self.CLASS_WEIGHTS,
        #         callbacks=callbacks_list
        #     )

        #     model.load_weights(os.path.join(save_dir, 'model_' + str(fold_var) + '.h5'))
        #     results = model.evaluate(x_val, y_val)
        #     results = dict(zip(model.metrics_names, results))

        #     val_acc.append(results['accuracy'])
        #     val_loss.append(results['loss'])
        #     val_f1.append(results['f1_m'])
        #     val_precision.append(results['precision_m'])
        #     val_recall.append(results['recall_m'])

        #     backend.clear_session()

        #     fold_var += 1

        # print('VAL_ACCURACY')
        # print(val_acc)
        # print(val_acc.mean())

        # print('VAL_LOSS')
        # print(val_loss)
        # print(val_loss.mean())

        # TRAIN MODEL
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            verbose=2,
            class_weight=self.CLASS_WEIGHTS,
            batch_size=64
        )

        log('Model training completed')
        log('Evaluating Model')

        # TEST MODEL
        y_test = np.array(y_val)
        y_scores = np.array(self.model.predict(x_val))

        # thresholds = np.arange(0, 1, 0.001)

        anx_thresh = 0.5
        dep_thresh = 0.5
        nan_thresh = 0.5

        for i, label in enumerate(self.LABELS):
            # ================
            # THRESHOLD MOVING
            # ================
            # scores = [f1_score(y_test[:, i], to_labels(y_scores[:, i], t))
            #           for t in thresholds]
            # ix = np.argmax(scores)
            # print('[%s] Threshold=%.3f, F-Score=%.5f' %
            #   (label, thresholds[ix], scores[ix]))

            # =========
            # PRC CURVE
            # =========
        #     precision, recall, thresholds = precision_recall_curve(
        #         y_test[:, i], y_scores[:, i])
        #     fscore = (2 * precision * recall) / \
        #         (precision + recall)
        #     ix = np.argmax(fscore)

        #     print('[%s] Best Threshold=%f, F-Score=%.3f' %
        #           (label, thresholds[ix], fscore[ix]))

        #     if i == 0:
        #         anx_thresh = thresholds[ix]
        #     elif i == 1:
        #         dep_thresh = thresholds[ix]
        #     elif i == 2:
        #         nan_thresh = thresholds[ix]


        #     plt.plot(recall, precision, marker='.', label='Logistic')
        #     plt.scatter(recall[ix], precision[ix],
        #                 marker='o', color='black', label='Best')

        # no_skill = len(y_test[y_test == 1]) / len(y_test)
        # plt.plot([0, 1], [no_skill, no_skill],
        #          linestyle='--', label='No Skill')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')

        # =========
        # ROC CURVE
        # =========
            fpr, tpr, thresh = roc_curve(
                y_test[:, i], y_scores[:, i])

            g_means = np.sqrt(tpr * (1 - fpr))
            ix = np.argmax(g_means)

            print('[%s] Best threshold=%f, G-MEAN=%.3f' %
                  (label, thresh[ix], g_means[ix]))

            plt.plot(fpr, tpr, marker='.', label=label)
            plt.scatter(fpr[ix], tpr[ix], marker='o',
                        color='black', label='Best')

        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend()
        plt.show()
        plt.close()


        y_pred = []

        for sample in y_scores:
            tmp = []
            tmp.append(1 if sample[0] > anx_thresh else 0)
            tmp.append(1 if sample[1] > dep_thresh else 0)
            tmp.append(1 if sample[2] > nan_thresh else 0)
            y_pred.append(tmp)

        y_pred = np.array(y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.LABELS
        )

        # GET MODEL'S STATISTICS
        loss, acc, _f1_score, precision, recall = self.model.evaluate(
            x_val, y_val)

        log('Saving model')

        # SAVE MODEL
        # self.model.save(self.model_file_path)

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

        for axes, cfs_matrix, label in zip(ax.flatten(), cm, ['Anxiety', 'Depression', 'None']):
            print_confusion_matrix(cfs_matrix, axes, label, ['N', 'Y'])

        cm_fig.tight_layout()
        plt.close()

        imgdata = StringIO()
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
                'f1_score': '{:.2f}'.format(_f1_score),
                'precision': '{:.2f}'.format(precision),
                'recall': '{:.2f}'.format(recall)
            }
        }

    def test(self, test_seq):
        log('Classifying tweets')

        preds = self.model.predict(test_seq)

        log('Getting average prediction')
        pred_avg = np.mean(preds, axis=0)
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


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')
