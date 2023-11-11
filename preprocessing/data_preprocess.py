class DataColumnCreation:
    def __init__(self, data):
        self.data = data

    def create_column(self, col_name, calculation_func):
        self.data[col_name] = calculation_func(self.data)
        return self.data
    