import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_clean import DataCleaning
from preprocessing.data_preprocess import DataStringEdition as transtr
from preprocessing.data_preprocess import DataColumnCreation as col_create



# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")
public = pd.read_csv("datasets/public_processed.csv")
df = pd.concat([train,public])
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")
# df = pd.read_csv("datasets/df_fill.csv")


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



# 使用 字串代碼轉成數字，比較好處理
replaced_cols = ['chid', 'cano', 'mchno', 'acqic']
df[replaced_cols] = transtr(data=df).str_trans_num(data=df,columns=replaced_cols)

#先將object 轉float
df[['etymd','stocn','scity']] = df[['etymd','stocn','scity']].astype(float)

#排序
df = df.sort_values(['locdt','hrs_loctm','chid','cano'])

def process_tw_scity(data, proportion = 0.9):
    # 先抓台灣training資料
    train_tw = data[(data.stocn==0)&(data.label.isin([0,1]))] 
    cum_fraud_tw = ((train_tw.groupby('scity')['label'].sum())/sum(train_tw.label)).sort_values(ascending=False).cumsum()
    # 取累積比例 > proportion 的 index
    twcity_others = set((data[(data.stocn==0)].scity.unique())).difference((cum_fraud_tw[cum_fraud_tw<proportion].index)) #TW所有city - TW熱門city
    data['new_scity'] = data['scity'].copy()
    condition = (data.stocn==0) & (data.scity.isin(twcity_others))
    data.loc[condition, 'new_scity'] = -1


df['label'] = df.label.astype(int)
# Functions #之後再丟改preprocess啦
# 類別資料
def latfeature_cuncount(data:pd.DataFrame, column, feat, colname:str, shift:int, start=0):
    data[colname] = -1
    for t in range(start, max(data.locdt)+1):
        if (t%7==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
        time_intervel = (data.locdt>=(t-shift+1))&(data.locdt<=t)
        sub_data = data[time_intervel][['locdt', column, feat]]
        sub_result = (sub_data.groupby(column)[feat].cumcount()+1)[sub_data.locdt==t].values
        data.loc[data['locdt'] == t, colname] = sub_result



def custom_mode(x):
    if (x.nunique() == 1): return x.iloc[0] 
    else: return (x.mode().iloc[0])
def latfeature_mode(data:pd.DataFrame, column, feat, colname:str, shift:int):
    data[colname] = -1
    for t in range(max(data.locdt)+1):
        if (t%8==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
        time_intervel = (data.locdt>=(t-shift+1))&(data.locdt<=t)
        sub_data = data[time_intervel][['locdt', column, feat]]
        sub_result = (sub_data.groupby(column)[feat].agg(lambda x: x.mode().iloc[0])).fillna(-1).values
        sub_result = (sub_data.groupby(column)[feat].agg(custom_mode)).fillna(-1).values
        
        
        data.loc[data['locdt'] == t, colname] = sub_result


# 數值型
def log1p_feature(data:pd.DataFrame, column):
    data[[f'{x}_log1p' for x in column]] = np.log1p(data[column])

def latfeature_mean(data:pd.DataFrame, column, feat, colname:str, shift:int, start=0):
    data[colname] = -1
    data[colname] = data[colname].astype(float)
    for t in range(start, max(data.locdt)+1):
        if (t%7==0) : print(f'{max(0,t-shift+1)}<=locdt<={t}')
        time_intervel = (data.locdt>=(t-shift+1))&(data.locdt<=t)
        sub_data = data[time_intervel][['locdt', column, feat]]
        sub_result = (sub_data.groupby(column)[feat].expanding().mean().values)[sub_data.locdt==t]
        data.loc[data['locdt'] == t, colname] = sub_result



# 把stocn的冷門國家換成-1(others)
stocn_most = (df.stocn.value_counts())[(df.stocn.value_counts()>10000)].index
df['new_stocn'] = np.where(df['stocn'].isin(stocn_most),df['stocn'], -1)

# 減少scity數量(台灣取累計盜刷比例>0.9的city換成other(-1))
process_tw_scity(df)

## 加變數
# 時間變數(周時分秒)
col_create(data=df).create_time()

# 發生日期間隔
df['diff_locdt'] = df.groupby('cano')['locdt'].diff().fillna(-1)

# 0是實體交易，1是線上+沒3D驗證，2是線上+3D驗證
df['ecfg_3dsmk'] = df['ecfg'] + df['flg_3dsmk'] 

# chid、cano分組近7天累積次數、ratio
latfeature_cuncount(df,column='cano',feat='txkey',colname='cano_cumcount',shift=7)
latfeature_cuncount(df,column='chid',feat='txkey',colname='chid_cumcount',shift=7)
df['cards_ratio'] = (df.cano_cumcount/df.chid_cumcount)

#log(x+1) + difference
log1p_feature(df,['conam','flam1','csmam'])
df['difflog1p_flam1conam'] = df['conam_log1p'] - df['flam1_log1p']
latfeature_mean(df, column='cano', feat='flam1_log1p', colname='flam1avg7log1p_cano', shift=7)
latfeature_mean(df, column='mcc', feat='flam1_log1p', colname='flam1avg7log1p_mcc', shift=7)
log1p_feature(df,['flam1avg7_cano']) 
df['flam1_diff_avg7log1p'] = df['flam1_log1p'] - df['flam1avg7log1p_cano']



# 將指定的類別變數轉換為物件型別
cat_cols1 = data_info[data_info['資料格式']=='類別型']['訓練資料欄位名稱'].iloc[:-2]
cat_cols2 = ['ecfg_3dsmk','new_stocn','new_scity','weekday']

df[cat_cols1] = df[cat_cols1].astype('object')
df[cat_cols2] = df[cat_cols2].astype('object')

new_train  = train[['txkey']].merge(df, how='left', on='txkey')
new_public = public[['txkey']].merge(df, how='left', on='txkey')

new_train.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm'], axis=1,inplace=True)
new_public.drop(['loctm','stocn','scity','flg_3dsmk','m_loctm','s_loctm','label'], axis=1,inplace=True)

# df.to_csv("datasets/df_fill.csv",index=False)
new_train.to_csv("datasets/new_train.csv",index=False)
new_public.to_csv("datasets/new_public.csv",index=False)
