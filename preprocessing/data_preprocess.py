class DataCleaning:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        cleaned_data = self.data.dropna()  # Remove rows with null values
        return cleaned_data

class DataColumnCreation:
    def __init__(self, data):
        self.data = data

    def create_column(self, col_name, calculation_func):
        self.data[col_name] = calculation_func(self.data)
        return self.data
    