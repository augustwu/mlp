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

class SplitHandler(tornado.web.RequestHandler):

    def get(self):
        preview = 5
        test_size = 0.2
        if inWeb():
            data = hdfs_read(q('path'))
            preview = int(q('preview'))
            test_size = float(q('ratio'))

        train_df, test_df = train_test_split(data, test_size=test_size)

        X_train = train_df.loc[:, "radius_mean":]
        y_train = train_df["diagnosis"]

        X_test = test_df.loc[:, "radius_mean":]
        y_test = test_df["diagnosis"]

        if inWeb():
            demo_x_train = 'demo_x_train_{}.csv'.format(int(time.time()))
            demo_y_train = 'demo_y_train_{}.csv'.format(int(time.time()))
            demo_x_test = 'demo_x_test_{}.csv'.format(int(time.time()))
            demo_y_test = 'demo_y_test_{}.csv'.format(int(time.time()))
            hdfs_write(X_train, demo_x_train)
            hdfs_write(y_train, demo_y_train)
            hdfs_write(X_test, demo_x_test)
            hdfs_write(y_test, demo_y_test)

            print(json.dumps({'X_train': demo_x_train, 'y_train': demo_y_train, 'X_test': demo_x_test,
                              'y_test': demo_y_test, 'X_train_preview': df2json(X_train),
                              'y_train_preview': df2json(y_train), 'X_test_preview': df2json(X_test),
                              'y_test_preview': df2json(y_test)}))
