# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances

# public  = pd.read_csv('./31_dataset_1st_training_public testing/dataset_1st/public_processed.csv')
# train  = pd.read_csv('./31_dataset_1st_training_public testing/dataset_1st/training.csv')

# df = pd.concat([public,train])
# df = df.fillna(-1)
# df.head(10)

# del public, train
# #把chid換成沒有重複的chid
# df['chid'] = df['cano'].map(df.groupby('cano')['chid'].first())
# #增加時間變數
# df['weekday'] = df.locdt % 7
# df['h_loctm'] = df.loctm // 10000
# df['m_loctm'] = (df.loctm % 10000) //100
# df['s_loctm'] = df.loctm % 100





#這邊最後再動
class DataColumnCreation:
    def __init__(self, data):
        self.data = data

    
    def create_time(self):
        
        #增加時間變數
        self.data['weekday'] = self.data.locdt % 7
        self.data['h_loctm'] = self.data.loctm // 10000
        self.data['m_loctm'] = (self.data.loctm % 10000) //100
        self.data['s_loctm'] = self.data.loctm % 100
        
        # loctm轉hrs
        self.data['hrs_loctm'] =  self.data.h_loctm  + self.data.m_loctm/60 + self.data.s_loctm/3600

        return self.data
    
    def moving_average(self, col_name: str, window_size: int):
        return self.data[col_name].rolling(window=window_size).mean().shift(1)


    # def create_column(self, col_name:list, calculation_func):
    #     self.data[col_name] = calculation_func(self.data)
    
    # 類別型
    def latfeature_cumcount(self, column, feat, colname:str, shift:int, start=0):
        if  colname not in self.data.columns:
            self.data[colname] = -1.0
        for t in range(start, max(self.data.locdt)+1):
            if (t%7==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
            time_intervel = (self.data.locdt>=(t-shift+1))&(self.data.locdt<=t)
            # sub_data = self.data[time_intervel][['locdt', column, feat]]
            # sub_result = (sub_data.groupby(column)[feat].cumcount()+1)[sub_data.locdt==t].values
            # self.data.loc[self.data['locdt'] == t, colname] = sub_result
            sub_data = self.data[time_intervel][['locdt', column, feat, colname]]
            grouped_cumcount = (sub_data.groupby(column)[feat].cumcount()+1)
            sub_data[colname][grouped_cumcount.index] = grouped_cumcount.values
            self.data.loc[self.data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
        return self.data



    def latfeature_nunique(self, column, feat, colname:str, shift:int, start=0):
        if  colname not in self.data.columns:
            self.data[colname] = -1.0        
        for t in range(start, max(self.data.locdt)+1):
            if (t%7==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
            time_intervel = (self.data.locdt>=(t-shift+1))&(self.data.locdt<=t)
            # sub_data = self.data[time_intervel][['locdt', column, feat]]
            # sub_result = (sub_data.groupby(column)[feat].cumcount()+1)[sub_data.locdt==t].values
            # self.data.loc[self.data['locdt'] == t, colname] = sub_result
            sub_data = self.data[time_intervel][['locdt', column, feat, colname]]
            grouped_cumcount = (sub_data.groupby(column)[feat].nunique())
            sub_data[colname][grouped_cumcount.index] = grouped_cumcount.values
            self.data.loc[self.data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
        return self.data

    # 數值型
    def log1p_feature(self, column):
        self.data[[f'{x}_log1p' for x in column]] = np.log1p(self.data[column])
        return self.data

    def latfeature_mean(self, column, feat, colname:str, shift:int, start=0):
        if  colname not in self.data.columns:
            self.data[colname] = -1.0
        for t in range(start, max(self.data.locdt)+1):
            if (t%7==0) : print(f'{max(0,t-shift+1)}<=locdt<={t}')
            time_intervel = (self.data.locdt>=(t-shift+1))&(self.data.locdt<=t)
            sub_data = self.data[time_intervel][['locdt', column, feat, colname]]
            grouped_mean = sub_data[[column, feat]].groupby(column)[feat].expanding().mean().reset_index(level=0, drop=True)
            sub_data[colname][grouped_mean.index] = grouped_mean.values
            self.data.loc[self.data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
        return self.data

    def latfeature_mode(self, column, feat, colname:str, shift:int):
        if  colname not in self.data.columns:
            self.data[colname] = -1.0        
        for t in range(max(self.data.locdt)+1):
            if (t%8==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
            time_intervel = (self.data.locdt>=(t-shift+1))&(self.data.locdt<=t)
            sub_data = self.data[time_intervel][['locdt', column, feat, colname]]
            grouped_mode = sub_data[[column, feat]].groupby(column)[feat].agg(lambda x: x.mode().iloc[0]).reset_index(level=0, drop=True)
            sub_data[colname][grouped_mode.index] = grouped_mode.values
            self.data.loc[self.data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
        return self.data
    
class FeatureEdition:
    def __init__(self, data, data_info):
        self.data = data
        self.mapping_dict = {}
        # self.new_data = None
        self.reversed_mapping_list = []
        self.data_info = data_info

    def chid_merge(self):
        #合併同樣cano下不同的chid
        self.data['chid'] = self.data['cano'].map(self.data.groupby('cano')['chid'].first())
        return self.data


    def str_trans_num(self, columns:list[str]):
        self.new_data = self.data.copy()  # 創建一個新的 DataFrame，以免影響原始資料

        for x in columns:
            for i, str_val in enumerate(self.data[x].unique()):
                self.mapping_dict[str_val] = i
            self.new_data[x] = self.data[x].map(self.mapping_dict)
        return self.new_data

    #把冷門國家換成-1
    def trans_stocn(self):
        stocn_most = (self.data.stocn.value_counts())[(self.data.stocn.value_counts()>10000)].index
        self.data['new_stocn'] = np.where(self.data['stocn'].isin(stocn_most),self.data['stocn'], -1)
        return self.data

    def process_tw_scity(self, proportion = 0.9):
    # 先抓台灣training資料
        train_tw = self.data[(self.data.stocn==0)&(self.data.label.isin([0,1]))] 
        cum_fraud_tw = ((train_tw.groupby('scity')['label'].sum())/sum(train_tw.label)).sort_values(ascending=False).cumsum()
        # 取累積比例 > proportion 的 index
        twcity_others = set((self.data[(self.data.stocn==0)].scity.unique())).difference((cum_fraud_tw[cum_fraud_tw<proportion].index)) #TW所有city - TW熱門city
        self.data['new_scity'] = self.data['scity'].copy()
        condition = (self.data.stocn==0) & (self.data.scity.isin(twcity_others))
        self.data.loc[condition, 'new_scity'] = -1
        return self.data

    def trans_cata2objcet(self, new_feat_trans2obj):
        cat_cols = self.data_info[self.data_info['資料格式']=='類別型']['訓練資料欄位名稱'].iloc[:-2]
        new_feat_trans2obj = ['ecfg_3dsmk','new_stocn','new_scity','weekday']

        self.data[cat_cols] = self.data[cat_cols].astype('object')
        self.data[new_feat_trans2obj] = self.data[new_feat_trans2obj].astype('object')
        return self.data


