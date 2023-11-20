import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from scipy.stats import randint, uniform
from preprocessing.data_preprocess import DataColumnCreation as col_create
from preprocessing.data_preprocess import FeatureEdition
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import os
# from keras.layers import Input, Dense
# from keras.models import Model
#Visualization
import matplotlib.pyplot as plt

# 加載數據
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
data = pd.read_csv('datasets/new_train.csv')

# A = data[data.new_stocn!=0].groupby('mcc')['label'].agg(['sum','mean','count'])


#Preprocess
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity','h_loctm','locdt'] #國外資料要丟insfg bnsfh iterm flbmk
drop_column2 = ['insfg','bnsfg','iterm','flbmk','ovrlt','ecfg']
data.drop(drop_column,axis=1,inplace=True)
data = data.sort_values(['time'])

# data[(data.new_stocn==0)].groupby('contp')['label'].agg(['sum','mean','count'])

# Data split
X = data[data.new_stocn==0]
y = data[data.new_stocn==0]['label']
X = X.drop(['time','label'], axis=1)
#Stocn ==0 不用拿掉

# X = X.drop(drop_column2, axis=1)

####################################################################################
#Model
# 定义LightGBM模型
#Cross Validation
# tscv = TimeSeriesSplit(n_splits=14)
kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

lgb_model = lgb.LGBMClassifier(objective='binary', random_state=41)
param_dist = {
    'classifier__num_leaves': randint(75, 125),
    'classifier__learning_rate': uniform(0.05,0.2),
    'classifier__max_depth': randint(8, 12),
    'classifier__min_child_samples': randint(80, 120),
    'classifier__subsample': uniform(0.9, 1.0 - 0.9),
    'classifier__colsample_bytree': [0.8, 0.9],
    'classifier__scale_pos_weight': [1, 3],
    'classifier__reg_alpha': [0, 3],
    'classifier__reg_lambda': [0, 3],
}

#　# 创建RandomUnderSampler对象

# Random Search
undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, scoring='f1', cv=kf, verbose=1, n_jobs=-1, random_state=42)

# 执行Random Search
random_search.fit(X, y)
best_model = random_search.best_estimator_
# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))

feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns
# 将特征重要性和对应的特征名字放在一起，并按重要性降序排序
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)


# 画直方图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df[f'Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

#####################################################
#Prediction
public = pd.read_csv('datasets/new_public.csv')
txkey_public = public['txkey']

# df = pd.read_csv('datasets/df_fill.csv')
# creat_feat  = col_create(df)
# trans_feat  = FeatureEdition( df, data_info)
# df = creat_feat.latfeature_cumcount(column='cano',feat='txkey',colname='cano_cumcount4',shift=4,start=49)
# df = creat_feat.latfeature_cumcount(column='chid',feat='txkey',colname='chid_cumcount4',shift=4,start=49)
# df = creat_feat.latfeature_mean(column='cano', feat='flam1_log1p', colname='flam1avg4_log1p_cano', shift=4,start=49)
# df = creat_feat.latfeature_mean(column='mcc', feat='flam1_log1p', colname='flam1avg4_log1p_mcc', shift=4,start=49)
# df['flam1_diff_avg4log1p_cano'] = df['flam1_log1p'] - df['flam1avg4_log1p_cano']
# df['flam1_diff_avg4log1p_mcc'] = df['flam1_log1p'] - df['flam1avg4_log1p_mcc']
# df['label_ratio_df_log1p'] = np.log1p(df.label_ratio_df)

# new_public = public[['txkey']].merge(df, how='left', on='txkey')
# new_public.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm','label'], axis=1,inplace=True)
# new_public.drop(drop_column,axis=1,inplace=True)
# new_public.drop(drop_column2,axis=1,inplace=True)
# public = new_public[X.columns]


public.drop(drop_column,axis=1,inplace=True)
new_predictions = best_model.predict(public)


#Output CSV
result_df = pd.DataFrame({'txkey': txkey_public, 'pred': new_predictions})
# Convert "txkey" to string (if it's not already)
result_df['txkey'] = result_df['txkey'].astype(str)
# Export the DataFrame to a CSV file
result_df.to_csv('datasets/public_prediction.csv', index=False)
