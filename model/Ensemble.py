import wittgenstein as lw
import pandas as pd

data = pd.read_csv('datasets/new_train.csv')
X = data.drop(columns=['label'])  # 假設'label'是目標變數
y = data['label']

# 初始化Ripper模型
ripper_clf = lw.RIPPER()

# 擬合模型
ripper_clf.fit(X, y)

# 打印規則
print(ripper_clf.ruleset_)
