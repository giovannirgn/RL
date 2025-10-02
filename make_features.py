import os
from paths import dataset_eur_usd,path_drive
from pandas import pandas
from utils_dataset import get_dataset






df = get_dataset(dataset_eur_usd)

df["Day"] = df.index.dayofweek
df["Hour"] = df.index.hour
df["pct_change_BidClose"]  = df["BidClose"].pct_change()
df["velocity_fourier_1H"]  = df["four_transf_1H_top_1"] - df["four_transf_1H_top_1"].shift(1)
df["velocity_fourier_1G"]  = df["four_transf_1G_top_1"] - df["four_transf_1G_top_1"].shift(1)
df["velocity_fourier_1M"]  = df["four_transf_1M_top_1"] - df["four_transf_1M_top_1"].shift(1)
df["diff_price_fourier_1H"] = df["BidClose"] - df["four_transf_1H_top_1"]
df["diff_price_fourier_1G"] = df["BidClose"] - df["four_transf_1G_top_1"]
df["diff_price_fourier_1M"] = df["BidClose"] - df["four_transf_1M_top_1"]

subs = df[['BidClose','sma_5min',
           'Day',
           'Hour',
           'pct_change_BidClose',
           'velocity_fourier_1H',
           'velocity_fourier_1G',
           'velocity_fourier_1M',
           'diff_price_fourier_1H',
           'diff_price_fourier_1G',
           "diff_price_fourier_1M"]]

subs.to_pickle(os.path.join(path_drive,"DATASET_EUR_USD.pkl"))



#df.loc["2024-06-07"]