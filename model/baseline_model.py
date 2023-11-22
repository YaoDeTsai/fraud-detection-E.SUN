import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint, uniform
from preprocessing.data_preprocess import DataColumnCreation as col_create
from preprocessing.data_preprocess import FeatureEdition
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import time
import os
import matplotlib.pyplot as plt
import joblib


joblib.parallel_backend('threading', n_jobs=12)
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
data = pd.read_csv('datasets/new_train.csv')

# data[(data.new_stocn==0)].groupby('contp')['label'].agg(['sum','mean','count'])

#Preprocess
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity','h_loctm','locdt'] #國外資料要丟insfg bnsfh iterm flbmk
drop_column2 = ['insfg','bnsfg','iterm','flbmk','ovrlt','ecfg']
data.drop(drop_column,axis=1,inplace=True)
data = data.sort_values(['time'])

# Data split
subset = data[(data['new_stocn'] == 0) & (data['label'] == 0)]
# 抽取1/10的數據
sampled_subset = subset.sample(frac=0.05, random_state=32)
tw_sample = pd.concat([sampled_subset,data[(data.new_stocn==0)&(data.label==1)]])

new_data =  pd.concat([tw_sample,data[data.new_stocn!=0]]).sort_values('time')
del sampled_subset, subset, tw_sample

X = data[data.new_stocn!=0]
y = data[data.new_stocn!=0]['label']
X = X.drop(['time','label'], axis=1)
#Stocn ==0 時不用拿掉
# X = X.drop(drop_column2, axis=1)

####################################################################################
#Model
# 定义LightGBM模型
#Cross Validation
tscv = TimeSeriesSplit(n_splits=14)
# kf = StratifiedKFold(n_splits=14, shuffle=True, random_state=42)

lgb_model = lgb.LGBMClassifier(objective='binary', random_state=41, device='gpu')
param_dist = {
    'classifier__num_leaves': randint(80, 120),
    'classifier__learning_rate': uniform(0.05, 0.2),
    'classifier__max_depth': randint(8, 12),
    'classifier__min_child_samples': randint(90, 130),
    'classifier__subsample': 0.9,
    'classifier__colsample_bytree': [0.9],
    'classifier__scale_pos_weight': [1],
    'classifier__reg_alpha': [0],
    'classifier__reg_lambda': [0],
}

#　# 创建RandomUnderSampler对象
# Random Search
undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)

# 执行Random Search
start_time = time.time()
random_search.fit(X, y)
best_model = random_search.best_estimator_
end_time = time.time()
execution_time = end_time - start_time

print(f"Training time: {execution_time} seconds")
# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))

##########################################################
# Prediction
public = pd.read_csv('datasets/new_public.csv')
public_ans = pd.read_csv('datasets/public.csv',usecols=['txkey', 'label'])

txkey_public = public['txkey']
public.drop(drop_column,axis=1,inplace=True)
public.drop('time',axis=1,inplace=True)

new_predictions = best_model.predict(public)
result_df = pd.DataFrame({'txkey': txkey_public, 'pred': new_predictions})
# Convert "txkey" to string (if it's not already)
result_df['txkey'] = result_df['txkey'].astype(str)
# Export the DataFrame to a CSV file
# result_df.to_csv('datasets/public_prediction.csv', index=False)
# 對答案
# result_df = pd.read_csv('datasets/public_prediction_abroad.csv')
pred_merge = pd.merge(result_df,public_ans,on = 'txkey')
f1 = f1_score(pred_merge.pred, pred_merge.label)
print(f1)




#######################################################
# 画直方图

feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df[f'Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

#目前模型 只拿國外資料train
# X = data[data.new_stocn!=0]
# y = data[data.new_stocn!=0]['label']


# Best_params ={'classifier__colsample_bytree': 0.9, 'classifier__learning_rate': 0.14121399684340719, 'classifier__max_depth': 10, 'classifier__min_child_samples': 92, 'classifier__num_leaves': 116, 'classifier__reg_alpha': 0, 'classifier__reg_lambda': 0, 'classifier__scale_pos_weight': 1, 'classifier__subsample': 0.968030753858778}