import pandas as pd
from preprocessing.data_clean import DataCleaning 

# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")

data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")

# missing data cleaning class
cleaner = DataCleaning(train, data_info)

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

print(train.isna().sum())