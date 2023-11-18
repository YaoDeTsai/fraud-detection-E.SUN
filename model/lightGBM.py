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
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import os
#Visualization
import matplotlib.pyplot as plt

# 加載數據
data = pd.read_csv('datasets/new_train.csv')


#Preprocess
# 處理類別型特徵：轉換為數值型
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])
# # 處理 NaN 值：填充或刪除
# imputer = SimpleImputer(strategy='median')
# data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity']
data.drop(drop_column,axis=1,inplace=True)
data = data.sort_values(['locdt','hrs_loctm'])

# Data split
data_abroad = data[data.new_stocn!=0]
tscv = TimeSeriesSplit(n_splits=6)
# 分割特徵和標籤
X = data_abroad.drop('label', axis=1)
y = data_abroad['label']

#Model
# 定义LightGBM模型
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42)

# 定义参数网格
param_grid = {
    'classifier__num_leaves': [50, 100],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [7, 10],
    'classifier__min_child_samples': [50, 100],
    'classifier__subsample': [0.9, 1.0],
    'classifier__colsample_bytree': [0.8],
    'classifier__scale_pos_weight': [1, 10],  # (1 - 0.0037) / 0.0037 ≈ 268.7027
    'classifier__reg_alpha': [0, 10],
    'classifier__reg_lambda': [0, 10],
}

# 创建RandomUnderSampler对象
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

# 创建Pipeline，将欠采样与模型一起包装
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', lgb_model)])

# 创建GridSearchCV对象
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=tscv, verbose=1, n_jobs=-1)
# 执行Grid Search
grid_search.fit(X, y)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best F1-score found: {:.4f}".format(grid_search.best_score_))

# 获取最佳模型
best_model = grid_search.best_estimator_['classifier']

# 打印特征重要性
feature_importance = best_model.feature_importances_
feature_names = X.columns

# 将特征重要性和对应的特征名字放在一起，并按重要性降序排序
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# 画直方图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()


#Prediction
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
