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

class TrainHandler(tornado.web.RequestHandler):

    def get(self):
        max_iter = 200
        tol = 0.0001

        if inWeb():
            X_train = hdfs_read(q('X_train'))
            y_train = hdfs_read(q('y_train'))
            max_iter = int(q('max_iter'))
            tol = float(q('tol'))

        clf = LogisticRegression(max_iter=max_iter, tol=tol)
        clf.fit(X_train, y_train)

        if inWeb():
            model_path = 'model_{}.pkl'.format(int(time.time()))
            save_model(clf, model_path)
            print(json.dumps({'model': model_path}))

        clf.fit(X_train, y_train)

