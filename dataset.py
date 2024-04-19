import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_dataset():
    df = pd.read_csv("cleaned_shifted_data.csv")
    
    drop_cols = [0,1,2,12,14,16]
    drop_cols = df.columns[drop_cols]
    df.drop(drop_cols,axis=1,inplace=True)

    X = df.drop('AQI_calculated_shifted',axis = 1)
    y = df['AQI_calculated_shifted']

    return X,y

def load_dataset_features():
    df = pd.read_csv("cleaned_shifted_data.csv")
    
    oe = OneHotEncoder(sparse_output=False)
    encoded = oe.fit_transform(pd.DataFrame(df['Station']))
    one_hot_df = pd.DataFrame(encoded, columns=oe.get_feature_names_out(['Station']))
    df = pd.concat([df, one_hot_df], axis=1)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['dayofweek'] = df['Timestamp'].dt.day_of_week

    drop_cols = [0,1,2,12,14,16]
    drop_cols = df.columns[drop_cols]
    df.drop(drop_cols,axis=1,inplace=True)

    X = df.drop('AQI_calculated_shifted',axis = 1)
    y = df['AQI_calculated_shifted']

    return X,y