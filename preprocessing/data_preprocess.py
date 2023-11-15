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
    def __init__(self,data):
        self.data = data

    def create_time(self):
        #合併同樣cano下重複的chid
        self.data['chid'] = self.data['cano'].map(self.data.groupby('cano')['chid'].first())
        
        #增加時間變數
        self.data['weekday'] = self.data.locdt % 7
        self.data['h_loctm'] = self.data.loctm // 10000
        self.data['m_loctm'] = (self.data.loctm % 10000) //100
        self.data['s_loctm'] = self.data.loctm % 100
        
        # loctm轉hrs
        self.data['hrs_loctm'] =  self.data.h_loctm  + self.data.m_loctm/60 + self.data.s_loctm/3600

        return self.data
    
    def moving_average(self, data: pd.DataFrame, col_name: str, window_size: int):
        return data[col_name].rolling(window=window_size).mean().shift(1)


    def create_column(self, col_name:list, calculation_func):
        self.data[col_name] = calculation_func(self.data)

    
# 將String轉成Int (還有反向)
class DataStringEdition:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.mapping_dict = {}
        self.new_data = None
        self.reversed_mapping_list = []


    def str_trans_num(self, data:pd.DataFrame, columns:list[str], reverse = False):
        self.new_data = self.data[columns].copy()  # 創建一個新的 DataFrame，以免影響原始資料

        if not reverse:
            for x in columns:
                for i, str_val in enumerate(data[x].unique()):
                    self.mapping_dict[str_val] = i
                self.new_data[x] = data[x].map(self.mapping_dict)
            return self.new_data
        else:
            for x in columns:
                unique_values = self.data[x].unique()
                self.reversed_mapping_list.append({i: v for i, v in enumerate(unique_values)})

            return self.reversed_mapping_list
        
    def num_trans_str(self, replaced_cols, reversed_mapping_list):
        self.new_data = self.data[replaced_cols].copy()  # 創建一個新的 DataFrame，以免影響原始資料
        
        for i, x in enumerate(replaced_cols):
            reversed_mapping = reversed_mapping_list[i]
            self.new_data[x] = self.data[x].map(reversed_mapping)

        return self.new_data
