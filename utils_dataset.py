import os
import pandas as pd
from paths import path_drive

_dataset = None

def get_dataset(dataset):
    global _dataset
    if _dataset is None:
        _dataset = pd.read_pickle(os.path.join(path_drive, dataset))
    return _dataset

def get_day(df, day):
    # day is a string
    return df.loc[day]

def get_range_dates(df,start,end):
    #start end are strings
    return df.loc[start:end]