import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class DataCleaning:
    def __init__(self, data):
        self.data = data
    
    def fill_stscd_neg1(self: pd.DataFrame):
        # fill stscd's na with -1.0
        self.stscd.fillna(-1.0,inplace=True)

    def fill_mcc_neg1(self: pd.DataFrame):
        # fill mcc's na with -1.0
        self.mcc.fillna(-1.0,inplace=True)
    
    def fill_csmcu_or_hcefg_acqic(self: pd.DataFrame, cs_hc: str):
        # find the acqic with the number of csmcu larger than 1000
        acqic_1000: pd.series = self[self[cs_hc].isna()]["acqic"].value_counts()>1000
        acqic_1000: list  = acqic_1000[acqic_1000].index.tolist()

        # create the dict of acqic_id and replace value
        new_val_byacqic = {value: (index+1)*-1.0 for index, value in enumerate(acqic_1000)}

        # fill the csmcu by for loop
        for acqic_id, fill_value in new_val_byacqic.items():
            self.loc[self['acqic'] == acqic_id, cs_hc] = self.loc[self['acqic'] == acqic_id, cs_hc].fillna(fill_value)

        # fill the rest na with the next value of dict
        self[cs_hc].fillna(list(new_val_byacqic.values())[-1],inplace=True)

    def proportionXX_target_col_count(self: pd.DataFrame, target_col: str, prop: float):
        # calculate the cumulative sum of counts
        counts = self[target_col].value_counts()
        cumulative_sum = counts.cumsum()

        # find the index values that contribute to 90% of the counts
        threshold = prop * counts.sum()
        selected_indices = cumulative_sum[cumulative_sum <= threshold].index
        return selected_indices
    
    def rf_train_test(self: pd.DataFrame, data_info: pd.DataFrame, target_col: str, sample_frac: float,prop: float):
        train_rf = self

        # identify columns containing string values
        string_columns = train_rf.select_dtypes(include='object').columns
        # drop columns containing string values
        train_rf = train_rf.drop(string_columns, axis=1)

        # select the categorical
        cat_col = data_info[data_info["資料格式"] == "類別型"]["訓練資料欄位名稱"]
        cat_col = cat_col[cat_col != target_col]
        # drop the label
        cat_col = cat_col[cat_col != "label"]
        cat_col = cat_col[cat_col.isin(train_rf.columns)]

        # replace categorical variables with frequency encoding
        df_freq_encoded = train_rf.copy()
        for column in cat_col:
            frequency_map = train_rf[column].value_counts(normalize=True).to_dict()
            df_freq_encoded[column] = train_rf[column].map(frequency_map)

        # split the train test by target_col isna
        target_col_train = df_freq_encoded[df_freq_encoded[target_col].notna()]
        target_col_test = df_freq_encoded[df_freq_encoded[target_col].isna()]

        # select the train by contribute to XX% of the counts
        target_col_train = target_col_train[target_col_train[target_col].isin(DataCleaning.proportionXX_target_col_count(self, target_col, prop))]

        # subset the data by random with XX% of sample size
        target_col_train = target_col_train.sample(frac= sample_frac, random_state=1111)

        # split the X and y for train and test
        X_train = target_col_train.drop(target_col, axis=1)
        y_train =  target_col_train[target_col].astype(str)
        X_test = target_col_test.drop(target_col, axis=1)

        # handle missing values
        X_train.fillna(-2.0, inplace=True)
        X_test.fillna(-2.0, inplace=True)

        # standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train   

    def fill_scity_etymd_byrf(self: pd.DataFrame, data_info: pd.DataFrame, target_col: str, sample_frac: float,prop: float):
        # target_col is the col need to fillna
        # sample_frac is the float number of proportion to sample the train data to use in RF
        # prop is the float number of selecting the train by contribute to XX% of the counts of target col's kind

        # output the train and test set
        X_train_scaled, X_test_scaled, y_train = DataCleaning.rf_train_test(self, data_info, target_col, sample_frac, prop)

        # train a RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train_scaled, y_train)

        # use the trained model to predict missing values
        predicted_values = rf_classifier.predict(X_test_scaled)

        self.loc[self[target_col].notna(),target_col] = self.loc[self[target_col].notna(),target_col].astype(str)
        # fill in the missing values with the predicted values
        self.loc[self[target_col].isna(),target_col] = predicted_values
