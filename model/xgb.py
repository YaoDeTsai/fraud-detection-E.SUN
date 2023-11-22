import xgboost as xgb
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import time
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from preprocessing.data_preprocess import prediction

# Train
data = pd.read_csv('datasets/new_concat.csv')
# Public+Private (提交的時候兩個都要)
test_data = pd.read_csv('datasets/test_data.csv')

data = data.sort_values(['time'])
drop_column = ['txkey','chid','cano','mchno','acqic','new_scity','h_loctm','locdt']+['insfg','bnsfg','iterm','flbmk','ovrlt','ecfg','contp']
data.drop(drop_column,axis=1,inplace=True)
#Preprocess
# Data split # 台灣正常樣本抽取5%的數據
subset = data[(data['new_stocn'] == 0) & (data['label'] == 0)]
sampled_subset = subset.sample(frac=0.05, random_state=41)
tw_sample = pd.concat([sampled_subset,data[(data.new_stocn==0)&(data.label==1)]]).sort_values('time')
new_data =  pd.concat([tw_sample,data[data.new_stocn!=0]]).sort_values('time')
del sampled_subset, subset, tw_sample


# X,y as training data
X = new_data
y = new_data['label']
X = X.drop(['time','label'], axis=1)
# 设置欠采样策略
random_seed = 42
tscv = TimeSeriesSplit(n_splits=12)

# XGBoost模型替换LightGBM
xgb_model = XGBClassifier(objective='binary:logistic', random_state=random_seed, tree_method='hist',device='cuda')

# 定义参数分布
param_dist = {
    'classifier__max_depth': randint(7, 12),
    'classifier__learning_rate': uniform(0.1, 0.2),
    'classifier__n_estimators': randint(100, 150),  # 替换为n_estimators
    'classifier__min_child_weight': randint(70, 100),  # 替换为min_child_weight
    'classifier__subsample': [0.9],
    'classifier__colsample_bytree': [0.8, 0.9],
    'classifier__scale_pos_weight': [1, 3],
    'classifier__reg_alpha': [0],
    'classifier__reg_lambda': [0],
}

undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=random_seed)
pipeline = Pipeline(steps=[('undersampler', undersampler), ('classifier', xgb_model)])
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=30, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)

# 执行 RandomizedSearchCV
start_time = time.time()
random_search.fit(X, y)
best_model_xgb = random_search.best_estimator_
end_time = time.time()
execution_time = end_time - start_time

# 输出最佳参数
print(f"Training time: {execution_time} seconds")
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))

#  保存 XGB 模型
best_xgb_model = best_model_xgb.named_steps['classifier']
# best_xgb_model.save_model('Xgb_model.json')

# 加載xgb MODEL
# best_xgb_model = xgb.Booster(model_file='Xgb_model.json')

# 對答案 
# 55~59的public可以看f1-score
result_xgb = prediction.output_result(test_data.drop('label',axis=1),best_xgb_model,drop_column)

# result_xgb = output_result(test_data.drop('label',axis=1),best_xgb_model,drop_columns)
f1 = f1_score(result_xgb[test_data.label>=0].pred, test_data[test_data.label>=0].label)
print(f1)



# 看Feature importance
# feature_importance = best_model_xgb.named_steps['classifier'].feature_importances_
# feature_names = X.columns
# # 将特征重要性和对应的特征名字放在一起，并按重要性降序排序
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df[f'Importance'])
# plt.xlabel('Importance')
# plt.title('Feature Importance')
# plt.show()


