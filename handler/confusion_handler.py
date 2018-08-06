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

class ConfusionHandler(tornado.web.RequestHandler):

    def get(self):

        if inWeb():
            error_df = hdfs_read(q('path'))
            threshold = float(q('threshold'))

        LABELS = ["Normal", "Fraud"]
        y_pred = [1 if e > threshold else 0 for e in error_df.predictions.values]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        if inWeb():
            print(json.dumps({'confusion': plt2str(plt)}))
        else:
            plt.show()
