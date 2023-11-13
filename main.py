import pandas as pd
from preprocessing.data_clean import DataCleaning as clean

# read the CSV file using pandas
train = pd.read_csv("datasets/training.csv")

data_info = pd.read_excel("datasets/31_資料欄位說明.xlsx")

# fill missing value
clean.fill_csmcu_or_hcefg_acqic(train, cs_hc = "hcefg")
clean.fill_csmcu_or_hcefg_acqic(train, cs_hc = "csmcu")
clean.fill_stscd_neg1(train)
clean.fill_mcc_neg1(train)

# target_col is the col need to fillna
# sample_frac is the float number of proportion to sample the train data to use in RF
# prop is the float number of selecting the train by contribute to XX% of the counts of target col's kind
clean.fill_scity_etymd_byrf(train, data_info, "etymd", 0.2, 1.0)
clean.fill_scity_etymd_byrf(train, data_info, "scity", 0.2, 0.9)
clean.fill_scity_etymd_byrf(train, data_info, "stocn", 0.2, 1.0)

print(train.isna().sum())