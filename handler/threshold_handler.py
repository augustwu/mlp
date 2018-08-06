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

class ThresholdHandler(tornado.web.RequestHandler):

    def get(self):
        threshold = 0.2

        if inWeb():
            error_df = hdfs_read(q('path'))
            threshold = float(q('threshold'))

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

        if inWeb():
            print(json.dumps({'threshold': plt2str(plt)}))
        else:
            plt.show()


