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

class DataHandler(tornado.web.RequestHandler):

    def get(self):
        csv_file = os.path.join(os.getcwd(),"data.csv")
        preview = 5
        if inWeb():
            data = hdfs_read(q('path'))
            preview = int(q('preview'))
        else:
            data = pd.read_csv(csv_file)

        # https://github.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/raw/master/data.csv

        data = data.loc[:, "id":"fractal_dimension_worst"]
        data.loc[data["diagnosis"] == "M", "diagnosis"] = 1
        data.loc[data["diagnosis"] == "B", "diagnosis"] = 0

        if inWeb():
            filename = 'demo_{}.csv'.format(int(time.time()))
            hdfs_write(data, filename)
            print(json.dumps({'filename': filename, 'preview': df2json(data, preview)}))

        self.write(data.head().to_json())
