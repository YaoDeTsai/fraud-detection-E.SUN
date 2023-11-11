import pandas as pd

class DataCleaning:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        cleaned_data = self.data.dropna()  # Remove rows with null values
        return cleaned_data