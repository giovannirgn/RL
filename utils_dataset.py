import os
import numpy as np
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


def moving_avg_optimized(np_array, window):

    # kernel per la media mobile
    kernel = np.ones(window) / window

    # calcolo la convoluzione → molto più veloce
    avg = np.convolve(np_array, kernel, mode="valid")

    # aggiungo le medie parziali per le finestre iniziali
    start = [np.mean(np_array[:i]) for i in range(1, window)]

    return pd.Series(start + avg.tolist()).array


def FourierTransformRolling(np_array, window, top_n=1):

    n = len(np_array)

    approx_prices = np.zeros(n)

    for t in range(n):


        start = max(0, t - window + 1)  # usa solo dati passati fino a t
        window_data = np_array[start:t+1]

        fft_vals = np.fft.fft(window_data)
        amp = np.abs(fft_vals)

        if top_n < len(amp):
            indices = np.argpartition(amp, -top_n)[-top_n:]
        else:
            indices = np.arange(len(amp))

        fft_filtered = np.zeros_like(fft_vals, dtype=complex)
        fft_filtered[indices] = fft_vals[indices]
        window_approx = np.fft.ifft(fft_filtered).real

        approx_prices[t] = window_approx[-1]  # solo il valore corrente

        if t % 100000 == 0:
            print(f"Processed {t} data points")

    return approx_prices