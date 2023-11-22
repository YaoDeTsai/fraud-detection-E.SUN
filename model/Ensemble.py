from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

################################

best_xgb_model = xgb.XGBClassifier()
best_xgb_model.load_model('Xgb_model.json')  # 使用 load_model 载入模型

best_lgb_model = lgb.LGBMClassifier()
best_lgb_model.booster_ = lgb.Booster(model_file='lgb_model_0.64.txt')  # 使用 booster_ 属性加载 LightGBM 模型

# voting_clf = VotingClassifier(estimators=[('lgb', best_lgb_model), ('xgb', best_xgb_model), ('rf', best_lgb_model)], voting='soft')
voting_clf = VotingClassifier(estimators=[('lgb', best_lgb_model), ('xgb', best_xgb_model)], voting='soft')

# 进行交叉验证
f1_scorer = make_scorer(f1_score)

tscv = TimeSeriesSplit(n_splits=12)
scores = cross_val_score(voting_clf, X, y, cv=tscv, scoring=f1_scorer)
# 计算准确率
f1_avg = scores.mean()


###########################################
# 创建 VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('lgb', best_lgb_model),
    ('xgb', best_xgb_model)
    # ,('rf', best_rf_model),
], voting='soft')

# 使用时间序列交叉验证进行拟合
tscv = TimeSeriesSplit(n_splits=12)
predictions_train = cross_val_predict(voting_clf, X, y, cv=tscv)

# 评估训练集上的准确性
accuracy_train = accuracy_score(y, predictions_train)
print(f"Training Accuracy: {accuracy_train}")

# 对 private set 进行预测
predictions_private = voting_clf.predict(X_private)

# 你可以使用 predictions_private 进行进一步的分析或评估