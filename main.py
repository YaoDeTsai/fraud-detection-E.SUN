import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_clean import DataCleaning as clean
from preprocessing.data_preprocess import DataStringEdition as transtr
from preprocessing.data_preprocess import DataColumnCreation as col_create



# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")
public = pd.read_csv("datasets/public_processed.csv")
df = pd.concat([train,public])
data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")


# fill missing value
clean.fill_csmcu_or_hcefg_acqic(df, cs_hc = "hcefg")
clean.fill_csmcu_or_hcefg_acqic(df, cs_hc = "csmcu")
clean.fill_stscd_neg1(df)
clean.fill_mcc_neg1(df)

# target_col is the col need to fillna
# sample_frac is the float number of proportion to sample the train data to use in RF
# prop is the float number of selecting the train by contribute to XX% of the counts of target col's kind
clean.fill_scity_etymd_byrf(df, data_info, "etymd", 0.2, 1.0)
clean.fill_scity_etymd_byrf(df, data_info, "scity", 0.2, 0.9)
clean.fill_scity_etymd_byrf(df, data_info, "stocn", 0.2, 1.0)




# 使用 字串代碼轉成數字，比較好處理
replaced_cols = ['chid', 'cano', 'mchno', 'acqic']
df[replaced_cols] = transtr(data=df).str_trans_num(data=df,columns=replaced_cols)

# 加變數
# 時間變數(周時分秒)
col_create(data=df).create_time()

#排序
df = df.sort_values(['locdt','hrs_loctm','chid','cano'])

# 發生日期間隔
df['diff_locdt'] = df.groupby('cano')['locdt'].diff().fillna(-1)

# 0是實體交易，1是線上+沒3D驗證，2是線上+3D驗證
df['ecfg_3dsmk'] = df['ecfg'] + df['flg_3dsmk'] 

# 把stocn的冷門國家換成-1(others)
stocn_most = (df.stocn.value_counts())[(df.stocn.value_counts()>10000)].index
df['new_stocn'] = np.where(df['stocn'].isin(stocn_most),df['stocn'], -1)

# 減少scity數量(台灣取累計盜刷比例>0.9的city換成other(-1))

def tw_scity_process(data, proportion = 0.9):
    # 先抓台灣training資料
    train_tw = data[(data.stocn==0)&(data.label>=0)] 
    cum_fraud_tw = ((train_tw.groupby('scity')['label'].sum())/sum(train_tw.label)).sort_values(ascending=False).cumsum()
    # 取累積比例 > proportion 的 index
    twcity_others = set((data[(data.stocn==0)].scity.unique())).difference((cum_fraud_tw[cum_fraud_tw<proportion].index)) #TW所有city - TW熱門city
    data['new_scity'] = data['scity'].copy()
    condition = (data.stocn==0) & (data.scity.isin(twcity_others))
    data.loc[condition, 'new_scity'] = -1

tw_scity_process(df)

# 國外scity還沒改

# 未完

##########################################################################################################################################
## EDA細節
# 盜刷幾乎都發生在國外(台灣僅占7%)  # 所以按比例推算public盜刷資料應該約2500筆
# 或許可以把模型分為國內、國外兩種

# 盜刷約3成發生在第一筆交易，約5成5與發生在第一筆交易的同一天(增加diff_locdt變數)
(df[df.label==1].diff_locdt.value_counts(normalize=True).sort_index())

# 在線上交易中，沒有3D驗證盜刷筆數風險約5.7倍(增加ecfg_3dsmk變數)
df[(df.label!=-1)&(df.ecfg_3dsmk==0)].label.mean()
df[(df.label!=-1)&(df.ecfg_3dsmk==1)].label.mean()
df[(df.label!=-1)&(df.ecfg_3dsmk==2)].label.mean()

# 把stocn中個數小於10000的合併最好，因為這樣合併盜刷比例最高(從下面程式碼看出來的)
a=[]
for x in range(1,31):
    b = 8000+1000*x
    C = (df[df.stocn.isin((df.stocn.value_counts())[(df.stocn.value_counts()<b)].index)].label.value_counts())
    print(b,(C[0]+C[1]))
    a.append(round(100*C[1]/(C[0]+C[1]),6))
    print(round(100*C[1]/(C[0]+C[1]),6),'%')

#國外沒有分期付款 #也不能紅利交易
df[(df.label!=-1)&(df.stocn!=0)&(df.insfg==1)].label.value_counts(normalize=True)
df[(df.label!=-1)&(df.stocn!=0)&(df.bnsfg==1)].label.value_counts(normalize=True)
