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
from sklearn.model_selection import cross_val_score
from scipy.stats import randint, uniform
from preprocessing.data_preprocess import prediction
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import time

import matplotlib.pyplot as plt
################################
#設定成12核
# import joblib
# joblib.parallel_backend('threading', n_jobs=12)

################################
# 加載數據
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
# train = pd.read_csv('datasets/new_train.csv')
data = pd.read_csv('datasets/new_concat.csv')
# data[(data.new_stocn==0)].groupby('contp')['label'].agg(['sum','mean','count'])

#Preprocess
data = data.sort_values(['time'])
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity','h_loctm','locdt']+['insfg','bnsfg','iterm','flbmk','ovrlt','ecfg','contp']
data.drop(drop_column,axis=1,inplace=True)
# data = data[data.time<=56]
# Data split # stocn==0抽取1/10的數據
subset = data[(data['new_stocn'] == 0) & (data['label'] == 0)]
sampled_subset = subset.sample(frac=0.05, random_state=41)
tw_sample = pd.concat([sampled_subset,data[(data.new_stocn==0)&(data.label==1)]]).sort_values('time')
new_data =  pd.concat([tw_sample,data[data.new_stocn!=0]]).sort_values('time')
del sampled_subset, subset, tw_sample

X = new_data
y = new_data['label']
X = X.drop(['time','label'], axis=1)
#Stocn ==0 不用拿掉
# X = X.drop(drop_column3, axis=1)

####################################################################################
#Model
# 定义LightGBM模型
#Cross Validation
random_seed = 42
tscv = TimeSeriesSplit(n_splits=12)
# kf = StratifiedKFold(n_splits=14, shuffle=True, random_state=42)

lgb_model = lgb.LGBMClassifier(objective='binary', random_state=random_seed, device='gpu')

param_dist = {
    'classifier__num_leaves': randint(150, 200),
    'classifier__learning_rate': uniform(0.1, 0.2),
    'classifier__max_depth': randint(7, 12),
    'classifier__min_child_samples': randint(70, 100),
    'classifier__subsample': [0.9],
    'classifier__colsample_bytree': [0.8, 0.9],
    'classifier__scale_pos_weight': [1,3],
    'classifier__reg_alpha': [0],
    'classifier__reg_lambda': [0],
}
#　# 创建RandomUnderSampler对象
# Random Search
undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=random_seed)
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=30, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)

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

feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns
# 将特征重要性和对应的特征名字放在一起，并按重要性降序排序
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

best_lgb_model = best_model.named_steps['classifier']
# 保存 LightGBM 模型
# best_lgb_model.booster_.save_model('Best_models/lgb_model_0.64.txt')

# 画直方图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df[f'Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
###################################################
#Best parameter

# def remove_step_name(prefix, params):
#     A =  {key.split(prefix, 1)[-1] if key.startswith(prefix) else key: value for key, value in params.items()}
#     return A
# Best_parameters={'classifier__colsample_bytree': 0.9, 'classifier__learning_rate': 0.13503117489824895, 'classifier__max_depth': 9, 'classifier__min_child_samples': 131, 'classifier__num_leaves': 78, 'classifier__reg_alpha': 3, 'classifier__reg_lambda': 0, 'classifier__scale_pos_weight': 1, 'classifier__subsample': 0.9842284774594998}
# # 將最佳超參數應用於 LightGBM 模型
# lgb_model.set_params(**remove_step_name('classifier__', Best_parameters))
# bestpara_model = lgb_model

# #將最佳超參數應用於 LightGBM 模型
# lgb_model.set_params(**best_params)

# 在訓練集上進行模型訓練
# bestpara_model.fit(X, y)

# # 使用交叉驗證計算 F1-score
# f1_scores = cross_val_score(best_lgb_model, X, y, cv=tscv, scoring='f1', n_jobs=-1)

# # 輸出每一個分割的 F1-score
# print("F1-scores for each split:", f1_scores)

# # 輸出平均 F1-score
# print("Average F1-score:", f1_scores.mean())

#####################################################
#Prediction
# public = pd.read_csv('datasets/final_public.csv')
# private = pd.read_csv('datasets/new_private.csv')
# full_data = pd.read_csv('datasets/df_fill.csv')
# full_data.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm'], axis=1,inplace=True)
# data_exp = pd.read_csv("datasets/31_範例繳交檔案.csv")
# test_data = pd.merge(data_exp['txkey'],full_data,on='txkey')
# test_data.to_csv('datasets/test_data.csv', index=False)
test_data = pd.read_csv('datasets/test_data.csv')


# 重新加载 LightGBM 模型
# best_lgb_model = lgb.Booster(model_file='lgb_model_0.64.txt')

# 使用加载的模型进行预测
# def output_result(data,model,drop_column):
#     txkey_public = data['txkey']
#     data.drop(drop_column,axis=1,inplace=True)
#     data.drop('time',axis=1,inplace=True)
#     new_predictions = model.predict(data)
#     result_df = pd.DataFrame({'txkey': txkey_public, 'pred': new_predictions})
#     result_df['txkey'] = result_df['txkey'].astype(str)
#     return result_df
# result = output_result(test_data.drop('label',axis=1),best_lgb_model,drop_column)


best_lgb_model = lgb.Booster(model_file='lgb_model_0.64.txt')
result = prediction.output_result(test_data.drop('label',axis=1),best_lgb_model,drop_column)

#輸出data
# result.to_csv('datasets/handin_pred0.66.csv', index=False)

result.pred.sum()

#2184*1354321/600182=4928

# 對答案
f1 = f1_score(result[test_data.label>=0].pred, test_data[test_data.label>=0].label)
print(f1)
