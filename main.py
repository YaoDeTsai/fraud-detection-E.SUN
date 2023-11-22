import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_clean import DataCleaning
from preprocessing.data_preprocess import DataColumnCreation as col_create
from preprocessing.data_preprocess import FeatureEdition



# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")
public = pd.read_csv("datasets/public.csv")
private = pd.read_csv("datasets/private_1_processed.csv")
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")

df = pd.concat([train,public,private], ignore_index=True)

# df = pd.read_csv("datasets/df_fill.csv")
# df.fillna(-1)
df[['stscd','label']] = df[['stscd','label']].fillna(-1)
df[['stscd','label','hcefg','csmcu']] = df[['stscd','label','hcefg','csmcu']].fillna(-1)
df[['stocn','scity']] = df[['stocn','scity']].fillna(-2)


# 隨機填補缺失值
def fill_proportion(data,colname):
    np.random.seed(42)
    missing_indices = df[colname].isnull()
    category_proportions = df[colname].value_counts(normalize=True)
    data.loc[missing_indices, colname] = np.random.choice(category_proportions.index, size=missing_indices.sum(), p=category_proportions.values)
    return data

df = fill_proportion(df,'etymd')
df = fill_proportion(df,'mcc')

# # missing data cleaning class
# cleaner = DataCleaning(df, data_info)

# # fill the na by -1
# cleaner.fill_stscd_neg1()
# cleaner.fill_mcc_neg1()

# # fill the na by the group of acqic
# cleaner.fill_csmcu_or_hcefg_acqic(cs_hc = "hcefg")
# cleaner.fill_csmcu_or_hcefg_acqic(cs_hc = "csmcu")

# # target_col is the col need to fillna
# # sample_frac is the float number of proportion to sample the train data to use in RF
# # prop is the float number of selecting the train by contribute to XX% of the counts of target col's kind
# cleaner.fill_scity_etymd_byrf("etymd", 0.3, 1.0)
# cleaner.fill_scity_etymd_byrf("scity", 0.3, 0.9)
# cleaner.fill_scity_etymd_byrf("stocn", 0.3, 1.0)


# Preprecess
# 使用 字串代碼轉成數字，比較好處理
creat_feat  = col_create(df)
trans_feat  = FeatureEdition( df, data_info)

df = trans_feat.str_trans_num(columns=['chid', 'cano', 'mchno', 'acqic'])
df[['etymd','stocn','scity']] = df[['etymd','stocn','scity']].astype(float)
df['label'] = df.label.fillna(-1).astype(int)
df = df.sort_values(['time','chid','cano']).reset_index(drop=True)


#################################

# 把stocn的冷門國家換成-1(others)
txkey_stocn = df[df['stocn']==-2].txkey 

creat_feat  = col_create(df)
trans_feat  = FeatureEdition( df, data_info)

df = trans_feat.trans_stocn()
df.loc[df.txkey.isin(txkey_stocn),'new_stocn'] = -2
# 減少scity數量(台灣取累計盜刷比例>0.9的city換成other(-1))
df = trans_feat.process_tw_scity()
#chid_merge
df = trans_feat.chid_merge()


## 加變數
df = creat_feat.create_time() #日時分秒
df['time'] = df['locdt'] + df['hrs_loctm']/24
df['diff_time'] = df.groupby('cano')['time'].diff().fillna(-1)
df['ecfg_3dsmk'] = df['ecfg'] + df['flg_3dsmk']  # 0是實體交易，1是線上+沒3D驗證，2是線上+3D驗證


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

# chid、cano分組近4天累積次數
df = creat_feat.latfeature_mean(column='cano', feat='flam1_log1p', colname='flam1avg4_log1p_cano', shift=4)
df = creat_feat.latfeature_mean(column='mcc', feat='flam1_log1p', colname='flam1avg4_log1p_mcc', shift=4)
df['flam1_diff_avg4log1p_cano'] = df['flam1_log1p'] - df['flam1avg4_log1p_cano']
df['flam1_diff_avg4log1p_mcc'] = df['flam1_log1p'] - df['flam1avg4_log1p_mcc']

#MCC類別
df = creat_feat.latfeature_mcc(shift = 4, start=4)
df = creat_feat.mcc_count(column_name='mcc_cat_rank')
df = creat_feat.mcc_count(column_name='mcc_risklabel',islabel=True)

#當天交易次數
df['transactions_count'] = df.groupby(['cano', 'locdt'])['txkey'].transform('count')
df['scity_count'] = df.groupby(['cano', 'locdt'])['scity'].transform('nunique')
df['mcc_count'] = df.groupby(['cano', 'locdt'])['mcc'].transform('nunique')
df['stocn_count'] = df.groupby(['cano', 'locdt'])['stocn'].transform('nunique')


#cano_cumcounts處理 # 0.97以上的個數合併
cano_cumcounts = df[df.label==1].groupby('cano_cumcount7')['label'].sum().cumsum()
cumcount_merge_idx = cano_cumcounts[cano_cumcounts/len(df[df.label==1])>=0.97].idxmin()
df.loc[df.cano_cumcount7>=cumcount_merge_idx,['cano_cumcount7']] = cumcount_merge_idx
df.loc[df.chid_cumcount7>=cumcount_merge_idx,['chid_cumcount7']] = cumcount_merge_idx


#mcc_mode_cano7
# df = creat_feat.latfeature_mode(column = 'cano',feat='mcc',colname='mcc_mode_cano7',shift=7)
# df['is_identity_mcc'] = np.where(df['mcc_mode_cano7'] == df['mcc'], 0, 1)




# # Scity處理
city_total = df[df['label'] != -1].groupby(['new_stocn', 'new_scity']).agg(n_labels=('label', 'sum'),total_transactions=('txkey', 'count')).reset_index()
city_total['label_ratio'] = city_total['n_labels'] / city_total['total_transactions']
# # 按照新建的 'n_labels' 欄位降序排列，取前20名
# city_total= city_total.rename(columns={'scity': 'new_scity'})
top_cities = city_total.sort_values(by='n_labels', ascending=False)
top_cities['cumsum_label'] = top_cities['n_labels'].cumsum() / sum(top_cities['n_labels'])
top_cities['temp'] = top_cities['n_labels'] + top_cities['label_ratio']
top_cities = top_cities.sort_values('temp',ascending=False)
top_cities['rank_scity'] = top_cities['temp'].rank(method='dense',ascending=False)
B = top_cities[(top_cities.total_transactions<10)]
top_cities.loc[(top_cities.total_transactions<10),'label_ratio'] = B['n_labels'].sum() / B['total_transactions'].sum()
df = pd.merge(df, top_cities[['new_stocn', 'new_scity' , 'label_ratio', 'rank_scity']], on=['new_stocn', 'new_scity'], how='left', suffixes=('_scity', ''))
    # df[(df.label!=-1)&(df.rank_scity <= 100)].label.sum()/32029
df.loc[(df['rank_scity']>=100),['rank_scity'] ] = 100

df.loc[df['mcc_risklabel']>80,['mcc_risklabel'] ] = 80
df['mcc_risklabel'] = df['mcc_risklabel'].fillna(80)

df.loc[df['mcc_cat_rank']>80,['mcc_cat_rank'] ] = 80
df['mcc_cat_rank'] = df['mcc_cat_rank'].fillna(80)

################################################
# 輸出
# label_ratio_df 跟nlabels_ratio 有na
columns_to_fill = df.columns[(df.isna().sum()!=0)]
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())
# df.to_csv("datasets/df_fill.csv",index=False)
############################################
# df = pd.read_csv('datasets/df_fill.csv')
df.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm'], axis=1,inplace=True)
data_exp = pd.read_csv("datasets/31_範例繳交檔案.csv")
test_data = pd.merge(data_exp['txkey'],df,on='txkey')
test_data.to_csv('datasets/test_data.csv', index=False)

new_concat_train  =  pd.concat([train,public], ignore_index=True)[['txkey']].merge(df, how='left', on='txkey')
new_concat_train.to_csv("datasets/new_concat.csv",index=False)

# new_private = private[['txkey']].merge(df, how='left', on='txkey')
# new_private.to_csv("datasets/new_private.csv",index=False)

############################################################

