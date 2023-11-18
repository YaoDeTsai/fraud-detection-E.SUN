import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_clean import DataCleaning
from preprocessing.data_preprocess import DataColumnCreation as col_create
from preprocessing.data_preprocess import FeatureEdition



# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")
public = pd.read_csv("datasets/public_processed.csv")
df = pd.concat([train,public])
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
# df = pd.read_csv("datasets/df_fill.csv")
# df.fillna(-1)

# missing data cleaning class
cleaner = DataCleaning(df, data_info)

# fill the na by -1
cleaner.fill_stscd_neg1()
cleaner.fill_mcc_neg1()

# fill the na by the group of acqic
cleaner.fill_csmcu_or_hcefg_acqic(cs_hc = "hcefg")
cleaner.fill_csmcu_or_hcefg_acqic(cs_hc = "csmcu")

# target_col is the col need to fillna
# sample_frac is the float number of proportion to sample the train data to use in RF
# prop is the float number of selecting the train by contribute to XX% of the counts of target col's kind
cleaner.fill_scity_etymd_byrf("etymd", 0.3, 1.0)
cleaner.fill_scity_etymd_byrf("scity", 0.3, 0.9)
cleaner.fill_scity_etymd_byrf("stocn", 0.3, 1.0)


# Preprecess
# 使用 字串代碼轉成數字，比較好處理
creat_feat  = col_create(df)
trans_feat  = FeatureEdition( df, data_info)



df = trans_feat.str_trans_num(columns=['chid', 'cano', 'mchno', 'acqic'])
df[['etymd','stocn','scity']] = df[['etymd','stocn','scity']].astype(float)
df['label'] = df.label.fillna(-1).astype(int)
df = df.sort_values(['locdt','loctm','chid','cano'])

# 把stocn的冷門國家換成-1(others)
df = trans_feat.trans_stocn()
# 減少scity數量(台灣取累計盜刷比例>0.9的city換成other(-1))
df = trans_feat.process_tw_scity()
#chid_merge
df = trans_feat.chid_merge()



## 加變數
df['diff_locdt'] = df.groupby('cano')['locdt'].diff().fillna(-1)
df['ecfg_3dsmk'] = df['ecfg'] + df['flg_3dsmk']  # 0是實體交易，1是線上+沒3D驗證，2是線上+3D驗證
df = creat_feat.create_time() #日時分秒

# chid、cano分組近7天累積次數
df = creat_feat.latfeature_cumcount(column='cano',feat='txkey',colname='cano_cumcount7',shift=7)
df = creat_feat.latfeature_cumcount(column='chid',feat='txkey',colname='chid_cumcount7',shift=7)


#log(x+1)
df = creat_feat.log1p_feature(['conam','flam1','csmam'])
df = creat_feat.latfeature_mean(column='cano', feat='flam1_log1p', colname='flam1avg7_log1p_cano', shift=7)
df = creat_feat.latfeature_mean(column='mcc', feat='flam1_log1p', colname='flam1avg7_log1p_mcc', shift=7)

#diff & ratio
df['cano_ratio'] = (df.cano_cumcount7/df.chid_cumcount7)
df['flam1conam_diff_log1p'] = df['conam_log1p'] - df['flam1_log1p']
df['flam1_diff_avg7log1p_cano'] = df['flam1_log1p'] - df['flam1avg7_log1p_cano']
df['flam1_diff_avg7log1p_mcc'] = df['flam1_log1p'] - df['flam1avg7_log1p_mcc']


# df.to_csv("datasets/df_fill.csv",index=False)
# 將指定的類別變數轉換為物件型別
new_train  = train[['txkey']].merge(df, how='left', on='txkey')
new_public = public[['txkey']].merge(df, how='left', on='txkey')

new_train.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm'], axis=1,inplace=True)
new_public.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm','label'], axis=1,inplace=True)

new_train.to_csv("datasets/new_train.csv",index=False)
new_public.to_csv("datasets/new_public.csv",index=False)

# # 分别设置正常和欺诈样本的数量
# n_normal = 30000
# n_fraud = 111


# # 分别从正常和欺诈样本中抽取
# normal_samples = df[df.label!=-1][df[df.label!=-1]['label'] == 0].sample(n=n_normal, random_state=1)  # 随机种子为了复现结果
# fraud_samples = df[df.label!=-1][df[df.label!=-1]['label'] == 1].sample(n=n_fraud, random_state=1)

# # 合并抽取的样本
# sampled_data = pd.concat([normal_samples, fraud_samples])


# from preprocessing.data_preprocess import FeatureEdition
# sampled_data = pd.read_csv("datasets/train_sample.csv")
# data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")

# cat_cols = data_info[data_info['資料格式']=='類別型']['訓練資料欄位名稱'].iloc[:-2]
# columns_to_remove = ['loctm', 'stocn', 'scity', 'flg_3dsmk', 'm_loctm', 's_loctm']
# cat_cols = [col for col in cat_cols if col not in columns_to_remove]
# new_feat_trans2obj = ['ecfg_3dsmk','new_stocn','new_scity','weekday']
# sampled_data[cat_cols] = sampled_data[cat_cols].astype('object')
# sampled_data[new_feat_trans2obj] = sampled_data[new_feat_trans2obj].astype('object')

# trans_feat  = FeatureEdition( sampled_data, data_info)
# trans_feat.trans_cata2objcet(['ecfg_3dsmk','new_stocn','new_scity','weekday'])
# sampled_data.to_csv('datasets/train_sample.csv', index=False)