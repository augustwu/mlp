
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

class PredictHandler(tornado.web.RequestHandler):

    def get(self):

        if inWeb():
            clf = load_model(q('model_path'))
            X_test = hdfs_read(q('x_test'))
            y_test = hdfs_read(q('y_test'))['diagnosis']

        predictions = clf.predict_proba(X_test)[:, -1]
        error_df = pd.DataFrame({'predictions': predictions,
                                'true_class': y_test})
        info = error_df.describe()
        if inWeb():
            error_filename = 'demo_error_{}.csv'.format(int(time.time()))
            hdfs_write(error_df, error_filename)
            print(json.dumps({'error': error_filename, 'error_info': df2json(info, lines=100)}))

        error_df.describe()
