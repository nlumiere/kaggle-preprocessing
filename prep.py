import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

def prep(df_test, df_train, one_hot_num = 20, omit=[], dropna=False):
    train_cols = df_train.columns
    test_cols = df_test.columns

    cols_diff = list(set(train_cols) - set(test_cols))
    if len(test_cols) > len(train_cols) or len(cols_diff) == len(train_cols):
        return None

    key_cols = df_train[cols_diff]

    if dropna:
        df_train = df_train.dropna()

    keep_cols = []
    for col in test_cols:
        if col in omit:
            keep_cols.append(col)
            continue
        uniques = df_train[col].unique()
        if is_numeric_dtype(df_train[col]) and is_numeric_dtype(df_test[col]):
            keep_cols.append(col)
            df_test[col] -= min(df_test[col])
            df_train[col] -= min(df_test[col])
            global_max = max(max(df_train[col]), max(df_test[col]))
            df_train[col]/=global_max
            df_test[col]/=global_max
        elif is_string_dtype(df_train[col]) and len(uniques) < one_hot_num:
            for e in uniques:
                keep_cols.append(e)
                df_train[e] = df_train[col] == e
                df_test[e] = df_test[col] == e
    df_train = df_train[keep_cols]
    df_test = df_test[keep_cols]

    return df_train, df_test, key_cols

def drop_cols(df, cols):
    cols_set = set(cols)
    df_cols_set = set(df.columns)
    if cols_set <= df_cols_set:
        return df[list(df_cols_set - cols_set)]
    
def split_cols(df, cols):
    cols_set = set(cols)
    df_cols_set = set(df.columns)
    if not cols_set <= df_cols_set:
        return df, None
    partition = df[cols]
    return df[list(df_cols_set - cols_set)], partition
    
