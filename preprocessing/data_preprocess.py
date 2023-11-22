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
    
    def mcc_count(self,column_name:str , islabel=False):
        data2 = self.data
        if islabel:  
            data2 = data2[data2.label==1]
        mcc_counts = data2.groupby(['new_stocn', 'mcc']).size().reset_index(name='count')
        mcc_counts[column_name] = mcc_counts.groupby('new_stocn')['count'].rank(ascending=False)
        self.data = pd.merge(self.data,mcc_counts[['new_stocn','mcc',column_name]], on=['new_stocn', 'mcc'], how='left')
        return self.data

    def latfeature_mode(self, column, feat, colname:str, shift:int):
        if  colname not in  self.data.columns:
            self.data[colname] = -1.0        
        for t in range(max( self.data.locdt)+1):
            if (t%8==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
            time_intervel = ( self.data.locdt>=(t-shift+1))&( self.data.locdt<=t)
            sub_data =  self.data[time_intervel][['locdt', column, feat, colname]]
            grouped_mode = sub_data[[column, feat]].groupby(column)[feat].agg(lambda x: x.mode().iloc[0]).reset_index(level=0, drop=True)
            common_index = sub_data[colname].index.intersection(grouped_mode.index)
            sub_data.loc[common_index, colname] = grouped_mode.loc[common_index].values
            # sub_data[colname][grouped_mode.index] = grouped_mode.values
            self.data.loc[ self.data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
        return  self.data
    
    def latfeature_mcc(self, shift:int, start=4):
        if 'label_ratio_mcc' not in self.data.columns:
            self.data[['label_ratio_mcc','nlabels_ratio']] = -1.0
        if (start - shift)<0 : start = shift
        for t in range(start, max(self.data.locdt)+1):
            if (start - shift)<0 : 
                print(t) 
                continue 
            if (t%15==14) : print(f't:{t} Range:{max(0,t-shift)}<=locdt<={t-1}')
            k = t
            if t>=(self.data[self.data.label==-1].locdt.min()) : t = (self.data[self.data.label!=-1].locdt.max()+1)
            time_intervel = (self.data.locdt>=(t-shift))&(self.data.locdt<=t-1)
            sub_data = self.data[time_intervel]
            if sub_data.empty: continue
            stocn_mcc_stats = (sub_data[sub_data.label!=-1].groupby(['new_stocn', 'mcc']).
                                agg(n_labels=('label', 'sum'), mcc_total=('label', 'count')).reset_index())
            # 计算比例
            stocn_mcc_stats['label_ratio_mcc'] = stocn_mcc_stats['n_labels'] / stocn_mcc_stats['mcc_total']
            stocn_mcc_stats['nlabels_ratio'] = stocn_mcc_stats['n_labels'] / stocn_mcc_stats.groupby('new_stocn')['n_labels'].transform('sum')
            result_data =  pd.merge(self.data[self.data['locdt'] == k][['new_stocn', 'mcc']], 
                                    stocn_mcc_stats[['new_stocn', 'mcc', 'label_ratio_mcc', 'nlabels_ratio']], on=['new_stocn', 'mcc'],
                                        how='left')
            self.data.loc[self.data['locdt'] == k, ['label_ratio_mcc', 'nlabels_ratio']] = result_data[['label_ratio_mcc', 'nlabels_ratio']].values

        self.data = self.data.fillna(-1.0)
        data_stocn_mcc = (self.data[self.data.label!=-1].groupby(['new_stocn', 'mcc']).agg(n_labels=('label', 'sum'), mcc_total=('label', 'count')).reset_index())
        data_stocn_mcc['label_ratio_mcc'] = data_stocn_mcc['n_labels'] / data_stocn_mcc['mcc_total']
        data_stocn_mcc['nlabels_ratio'] = data_stocn_mcc['n_labels'] / data_stocn_mcc.groupby('new_stocn')['n_labels'].transform('sum')

        left_missingdata =  pd.merge(self.data[(self.data.nlabels_ratio==-1)&(self.data.label_ratio_mcc==-1)][['new_stocn', 'mcc']], 
                                    data_stocn_mcc[['new_stocn', 'mcc', 'label_ratio_mcc', 'nlabels_ratio']], on=['new_stocn', 'mcc'],
                                        how='left')
        self.data.loc[(self.data['nlabels_ratio'] == -1) & (self.data['label_ratio_mcc'] == -1), ['label_ratio_mcc', 'nlabels_ratio']] = left_missingdata[['label_ratio_mcc', 'nlabels_ratio']].values
        self.data.loc[(self.data['nlabels_ratio'] == -1) & (self.data['label_ratio_mcc'] != -1), ['label_ratio_mcc', 'nlabels_ratio']] = 0

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

class prediction:
    # def __init__():
        
    def output_result(data, model, drop_column):
        txkey_public = data['txkey']
        data.drop(drop_column,axis=1,inplace=True)
        data.drop('time',axis=1,inplace=True)
        new_predictions = model.predict(data)
        result_df = pd.DataFrame({'txkey': txkey_public, 'pred': new_predictions})
        result_df['txkey'] = result_df['txkey'].astype(str)
        return result_df
