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
from scipy.stats import randint, uniform
from preprocessing.data_preprocess import DataColumnCreation as col_create
from preprocessing.data_preprocess import FeatureEdition
from imblearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

import os
#Visualization
import matplotlib.pyplot as plt
#ENN
from imblearn.under_sampling import EditedNearestNeighbours

# 加載數據
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
data = pd.read_csv('datasets/new_train.csv')
# creat_feat  = col_create(data)
# trans_feat  = FeatureEdition( data, data_info)
# data = creat_feat.latfeature_cumcount(column='cano',feat='txkey',colname='cano_cumcount4',shift=4)
# data = creat_feat.latfeature_cumcount(column='chid',feat='txkey',colname='chid_cumcount4',shift=4)
# data = creat_feat.latfeature_mean(column='cano', feat='flam1_log1p', colname='flam1avg4_log1p_cano', shift=4)
# data = creat_feat.latfeature_mean(column='mcc', feat='flam1_log1p', colname='flam1avg4_log1p_mcc', shift=4)
# data['flam1_diff_avg4log1p_cano'] = data['flam1_log1p'] - data['flam1avg4_log1p_cano']
# data['flam1_diff_avg4log1p_mcc'] = data['flam1_log1p'] - data['flam1avg4_log1p_mcc']

# data['label_ratio_df_log1p'] = np.log1p(data.label_ratio_df)

#Preprocess
# 處理類別型特徵：轉換為數值型
# label_encoders = {}
# categorical_columns = data.select_dtypes(include=['object']).columns
# for col in categorical_columns:
#     label_encoders[col] = LabelEncoder()
#     data[col] = label_encoders[col].fit_transform(data[col])
# # 處理 NaN 值：填充或刪除
# imputer = SimpleImputer(strategy='median')
# data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity','h_loctm','locdt','insfg','bnsfg','iterm','flbmk'] #國外資料要丟insfg bnsfh iterm flbmk
drop_column2 = ['ovrlt','ecfg']
data.drop(drop_column,axis=1,inplace=True)
data = data.sort_values(['time'])


# Data split
# data_abroad = data[data.new_stocn==0]
data_abroad = data

# data_abroad = data
tscv = TimeSeriesSplit(n_splits=8)
# 分割特徵和標籤
X = data_abroad.drop(['time','label'], axis=1)
X = X.drop(drop_column2, axis=1)
y = data_abroad['label']

#Model
# 定义LightGBM模型
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=41)

# # 定义参数网格
# 创建Pipeline，将欠采样与模型一起包装
# pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])


# # 创建GridSearchCV对象
# grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=tscv, verbose=1, n_jobs=-1)
# # 执行Grid Search
# grid_search.fit(X, y)
# # 输出最佳参数
# print("Best parameters found: ", grid_search.best_params_)
# print("Best F1-score found: {:.4f}".format(grid_search.best_score_))

# # 获取最佳模型
# best_model = grid_search.best_estimator_['classifier']
# # 打印特征重要性
# feature_importance = best_model.feature_importances_
# feature_names = X.columns


param_dist = {
    'classifier__num_leaves': randint(50, 100),
    'classifier__learning_rate': uniform(0.09, 0.1 - 0.05),
    'classifier__max_depth': randint(7, 10),
    'classifier__min_child_samples': randint(80, 120),
    'classifier__subsample': uniform(0.92, 1.0 - 0.92),
    'classifier__colsample_bytree': [0.8],
    'classifier__scale_pos_weight': [1, 5, 10],
    'classifier__reg_alpha': [0, 5, 10],
    'classifier__reg_lambda': [0, 5, 10],
}

#　# 创建RandomUnderSampler对象

# Random Search
undersampler = RandomUnderSampler(sampling_strategy=0.15, random_state=42)
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)
# 执行Random Search
random_search.fit(X, y)
best_model = random_search.best_estimator_
# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))

# 獲取特徵重要性
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

#Random Forest
# 定義你的模型，這裡以Random Forest為例
model = RandomForestClassifier(random_state=42)

# 定義超參數的範圍
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

# 創建RandomizedSearchCV對象
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,  # 設定搜尋的次數
    scoring='f1',  # 選擇合適的評估指標
    cv=tscv,  # 交叉驗證的折數
    verbose=1,
    n_jobs=-1,  # 使用所有可用的CPU核心進行搜尋
    random_state=42
)

# 執行Randomized Search
random_search.fit(X, y)

# 獲取最佳模型
best_model = random_search.best_estimator_

# 輸出最佳參數
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))


########################Prediction
public = pd.read_csv('datasets/new_public.csv')
txkey_public = public['txkey']

drop_column = ['txkey','chid','cano','mchno','acqic','new_scity']
public.drop(drop_column,axis=1,inplace=True)

new_predictions = best_model.predict(public)

#Output CSV
result_df = pd.DataFrame({'txkey': txkey_public, 'pred': new_predictions})
# Convert "txkey" to string (if it's not already)
result_df['txkey'] = result_df['txkey'].astype(str)
# Export the DataFrame to a CSV file
result_df.to_csv('datasets/public_prediction.csv', index=False)
