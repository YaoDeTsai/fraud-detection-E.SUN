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
df['time'] = df['locdt'] + df['hrs_loctm']/24
df['diff_time'] = df.groupby('cano')['time'].diff().fillna(-1)
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

# MCC特徵，我現在不敢亂動
def latfeature_mcc(data, shift:int, start=4):
    if 'label_ratio' not in data.columns:
        data[['label_ratio','nlabels_ratio']] = -1.0
    if (start - shift)<0 : start = shift
    for t in range(start, max(data.locdt)+1):
        if (start - shift)<0 : 
            print(t) 
            continue 
        if (t%15==14) : print(f't:{t} Range:{max(0,t-shift)}<=locdt<={t-1}')
        k = t
        if t>=(data[data.label==-1].locdt.min()) : t = (data[data.label!=-1].locdt.max()+1)
        time_intervel = (data.locdt>=(t-shift))&(data.locdt<=t-1)
        sub_data = data[time_intervel]
        if sub_data.empty: continue
        stocn_mcc_stats = (sub_data[sub_data.label!=-1].groupby(['new_stocn', 'mcc']).
                            agg(n_labels=('label', 'sum'), mcc_total=('label', 'count')).reset_index())
        # 计算比例
        stocn_mcc_stats['label_ratio'] = stocn_mcc_stats['n_labels'] / stocn_mcc_stats['mcc_total']
        stocn_mcc_stats['nlabels_ratio'] = stocn_mcc_stats['n_labels'] / stocn_mcc_stats.groupby('new_stocn')['n_labels'].transform('sum')
        result_data =  pd.merge(data[data['locdt'] == k][['new_stocn', 'mcc']], 
                                stocn_mcc_stats[['new_stocn', 'mcc', 'label_ratio', 'nlabels_ratio']], on=['new_stocn', 'mcc'],
                                    how='left')
        data.loc[data['locdt'] == k, ['label_ratio', 'nlabels_ratio']] = result_data[['label_ratio', 'nlabels_ratio']].values

    data = data.fillna(-1.0)
    data_stocn_mcc = (data[data.label!=-1].groupby(['new_stocn', 'mcc']).agg(n_labels=('label', 'sum'), mcc_total=('label', 'count')).reset_index())
    data_stocn_mcc['label_ratio'] = data_stocn_mcc['n_labels'] / data_stocn_mcc['mcc_total']
    data_stocn_mcc['nlabels_ratio'] = data_stocn_mcc['n_labels'] / data_stocn_mcc.groupby('new_stocn')['n_labels'].transform('sum')

    left_missingdata =  pd.merge(data[(data.nlabels_ratio==-1)&(data.label_ratio==-1)][['new_stocn', 'mcc']], 
                                data_stocn_mcc[['new_stocn', 'mcc', 'label_ratio', 'nlabels_ratio']], on=['new_stocn', 'mcc'],
                                    how='left')
    data.loc[(data['nlabels_ratio'] == -1) & (data['label_ratio'] == -1), ['label_ratio', 'nlabels_ratio']] = left_missingdata[['label_ratio', 'nlabels_ratio']].values
    data.loc[(data['nlabels_ratio'] == -1) & (data['label_ratio'] != -1), ['label_ratio', 'nlabels_ratio']] = 0

    return data

df = latfeature_mcc(df, shift=4,start =0)

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

# # Scity處理
city_total = df[df['label'] != -1].groupby(['new_stocn', 'new_scity']).agg(n_labels=('label', 'sum'),
                                                                       total_transactions=('txkey', 'count')).reset_index()
city_total['label_ratio'] = city_total['n_labels'] / city_total['total_transactions']
# # 按照新建的 'n_labels' 欄位降序排列，取前20名
# city_total= city_total.rename(columns={'scity': 'new_scity'})
top_cities = city_total.sort_values(by='n_labels', ascending=False)
top_cities['cumsum_label'] = top_cities['n_labels'].cumsum() / sum(top_cities['n_labels'])
top_cities['temp'] = top_cities['n_labels'] + top_cities['label_ratio']
top_cities = top_cities.sort_values('temp',ascending=False)
top_cities['rank_scity'] = top_cities['temp'].rank(method='dense',ascending=False)
A = top_cities[(top_cities.total_transactions<10)]
top_cities.loc[(top_cities.total_transactions<10),'label_ratio'] = A['n_labels'].sum() / A['total_transactions'].sum()
df = pd.merge(df, top_cities[['new_stocn', 'new_scity' , 'label_ratio', 'rank_scity']], on=['new_stocn', 'new_scity'], how='left')
# df[(df.label!=-1)&(df.rank_scity <= 100)].label.sum()/32029
df.loc[(df['rank_scity']>=100),['rank_scity'] ]= 100

# # del df['label_ratio_x'],df['rank_scity_x'],df['label_ratio_y'],df['rank_scity_y']


def latfeature_mode(data, column, feat, colname:str, shift:int):
    if  colname not in  data.columns:
         data[colname] = -1.0        
    for t in range(max( data.locdt)+1):
        if (t%8==0):print(f'{max(0,t-shift+1)}<=locdt<={t}')
        time_intervel = ( data.locdt>=(t-shift+1))&( data.locdt<=t)
        sub_data =  data[time_intervel][['locdt', column, feat, colname]]
        grouped_mode = sub_data[[column, feat]].groupby(column)[feat].agg(lambda x: x.mode().iloc[0]).reset_index(level=0, drop=True)
        common_index = sub_data[colname].index.intersection(grouped_mode.index)
        sub_data.loc[common_index, colname] = grouped_mode.loc[common_index].values
        # sub_data[colname][grouped_mode.index] = grouped_mode.values
        data.loc[ data['locdt'] == t, colname] = sub_data[sub_data.locdt == t][colname]
    return  data

A = latfeature_mode(df,'cano',feat='mcc',colname='mcc_mode_cano7',shift=7)


# label_ratio_df 跟nlabels_ratio 有na
columns_to_fill = df.columns[(df.isna().sum()!=0)]
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())
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