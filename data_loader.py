import pandas as pd

def load_and_prepare(file):
    df = pd.read_csv(file)
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    return df
