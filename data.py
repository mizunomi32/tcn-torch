#!/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.utils import shuffle

class Data:
    def __init__(self, wd=".", show_stdout=False):
        self.wd = wd
        self.show_stdout = show_stdout

    def _read_csv(self, file):
        data = pd.read_csv('%s' % file, header=None)
        a_data = data[[0, 1, 2]]  # axis of data
        a_data.columns = ['x_axis', 'y_axis', 'z_axis']
        return a_data

    def _read_data(self,class_num=19, data_num=480):
        feature = list()
        cls = sorted(os.listdir('%s/daily_and_sports_activities_dataset' % self.wd))
        # get people list
        pn = sorted(os.listdir('%s/daily_and_sports_activities_dataset/%s' % (self.wd, cls[0])))
        for k in range(len(cls[:class_num])):
            c_f = list()  # feature_extraction for each class
            if self.show_stdout:
                print("class %d start"%k)
            class_files = list()
            for j in range(len(pn)):
                files = glob.glob(
                    '%s/daily_and_sports_activities_dataset/%s/%s/a*'
                    % (self.wd, cls[k], pn[j])
                )
                class_files.extend(files)
            class_files = shuffle(class_files)
            #  クラスのファイル読み込みを並列化
            p = Pool(20)
            output = p.map(self._read_csv, class_files[:data_num])
            # プロセスの終了
            p.close()
            # store datas in list  #
            feature.append(output)
            if self.show_stdout:
                print("class %d end"%k)
        return feature

    def data(self, class_num=19, data_num=480):
        feature = self._read_data(class_num=class_num,data_num=data_num)
        x_raw = list()
        y_raw = list()
        for i in range(class_num):
            x_raw.extend([k.values for k in feature[i]])
            tmp = []
            for j in range(data_num):
                tmp.append(i)
            y_raw.extend(tmp)
        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        return x_raw, y_raw
