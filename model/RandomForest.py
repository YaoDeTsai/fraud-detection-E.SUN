from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from imblearn.pipeline import make_pipeline
import time
import pandas as pd
import joblib

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

random_seed = 42
undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=random_seed)

# 定义Random Forest模型
rf_model = RandomForestClassifier(random_state=random_seed)

# 定义参数分布
param_dist = {
    'randomforestclassifier__n_estimators': randint(100, 150),
    'randomforestclassifier__max_depth': randint(6, 15),
    'randomforestclassifier__min_samples_split': randint(2, 20),
    'randomforestclassifier__min_samples_leaf': randint(1, 20),
    'randomforestclassifier__bootstrap': [True, False],
    'randomforestclassifier__class_weight': ['balanced', 'balanced_subsample', None],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.5]
}

# Random Search
pipeline = make_pipeline(undersampler, rf_model)
tscv = TimeSeriesSplit(n_splits=7)
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=3, scoring='f1', cv=tscv, verbose=1, n_jobs=-1, random_state=42)

# 执行Random Search
start_time = time.time()

random_search.fit(X, y)
best_model_rf = random_search.best_estimator_

end_time = time.time()
execution_time = end_time - start_time

print(f"Training time: {execution_time} seconds")
# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)
print("Best F1-score found: {:.4f}".format(random_search.best_score_))


# 儲存模型
joblib.dump(best_model_rf, 'Best_models/rf_model.joblib')

# # 加载模型
# loaded_model = joblib.load('Best_models/rf_model.joblib')

# # 使用加载的模型进行预测
# predictions = loaded_model.predict(X_test)

# 對答案
# 55~59的public可以看f1-score
result_rf = output_result(test_data.drop('label',axis=1),best_model_rf,drop_columns)
f1 = f1_score(result_rf[test_data.label>=0].pred, test_data[test_data.label>=0].label)
print(f1)


