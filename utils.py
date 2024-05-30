# utils.py

import pandas as pd

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def split_data(df, train_idx, val_idx):
    X = df.iloc[:, 3:].values
    y = df.iloc[:, 2].values
    print('-'*50)
    print(f'Data loaded successfully! \n Details:')
    print(f'Shape of X: {X.shape}, Shape of y: {y.shape}')
    X_train, X_val, X_test = X[:int(train_idx)], X[int(train_idx):int(val_idx)], X[int(val_idx):]
    y_train, y_val, y_test = y[:int(train_idx)], y[int(train_idx):int(val_idx)], y[int(val_idx):]
    print(f'Shape of y_train: {y_train.shape}, Shape of y_val: {y_val.shape}, Shape of y_test: {y_test.shape}')
    return X_train, X_val, X_test, y_train, y_val, y_test
