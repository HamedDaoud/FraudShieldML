import numpy as np
import pandas as pd

def add_cyclic_time_features(df, time_col='Time'):
    hour = (df[time_col] // 3600) % 24
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return df

def log_transform_amount(df, amount_col='Amount'):
    df['LogAmount'] = np.log1p(df[amount_col])
    return df