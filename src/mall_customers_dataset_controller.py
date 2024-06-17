from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

class MallCustomersDatasetController:

    def __init__(self) -> None:
        self.df = pd.read_csv('dataset/Mall_Customers.csv', sep=',')
        self.df_ids = list(self.df.iloc[:, 0])
        self.df = self.df.drop('CustomerID', axis=1)
        self.df.columns = ['Genre', 'Age', 'Annual Income', 'Spending Score']

    # Print di alcune informazioni utili inerenti al dataset
    def print_column_info(self) -> None:
        print('*** Column Information ***')
        for column in self.df.columns:
            print(f'Column: {column}')
            print(f' - Data Type: {self.df[column].dtype}')
            print(f' - Missing Values: {self.df[column].isnull().sum()}')
            print()
    
    # Rimozione valori nulli
    def na(self) -> None:
        self.df = self.df.replace(np.nan, 0)
        print('N.A. data:\n', self.df.isnull().sum(), '\n')
    
    # Rimozione duplicati
    def remove_duplicated(self) -> None:
        self.df = self.df.drop_duplicates()
        print(f'There are {self.df.duplicated().sum()} duplicated instances.\n')

    # Label Encoding per la variabile categorica 'Genre'
    def label_encoding(self) -> None:
        le = LabelEncoder()
        self.df['Genre'] = le.fit_transform(self.df['Genre'])
        print('Label Encoding complete.\n')

    # Normalizzazione dei dati
    def normalize_data(self) -> None:
        scaler = MinMaxScaler()
        self.df[['Age', 'Annual Income', 'Spending Score']] = scaler.fit_transform(self.df[['Age', 'Annual Income', 'Spending Score']])
        print('Data normalization complete.\n')

    # Restituisce il dataframe
    def get_df(self) -> pd.DataFrame:
        return self.df
    
    # Restituisce il dataframe
    def get_df_ids(self) -> list:
        return self.df_ids
    
