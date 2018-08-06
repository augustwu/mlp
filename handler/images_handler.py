#coding=utf-8

import tornado
from tools.inweb import inWeb
import numpy as np
import pandas as pd
import time

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns

from hdfs import InsecureClient
import json
import pickle,os
import base64

class ImagesHandler(tornado.web.RequestHandler):

    def get(self):

        if inWeb():
            error_df = hdfs_read(q('path'))

        fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.predictions)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.001, 1])
        plt.ylim([0, 1.001])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        aoc = plt2str(plt)

        if not inWeb():
            plt.show()

        threshold = 0.2

        groups = error_df.groupby('true_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.predictions, marker='o', ms=1, linestyle='',
                    label= "Fraud" if name == 1 else "Normal")
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Error")
        plt.xlabel("Data point index")

        thres = plt2str(plt)

        if not inWeb():
            plt.show()

        LABELS = ["Normal", "Fraud"]
        y_pred = [1 if e > threshold else 0 for e in error_df.predictions.values]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        confusion = plt2str(plt)

        if inWeb():
            print(json.dumps({'threshold': thres, 'aoc': aoc, 'confusion': confusion}))
        else:
            plt.show()
