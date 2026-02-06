# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\final_processed_data_full_pipeline.csv')

const_cols = [c for c in df.columns if df[c].nunique() <= 1]
df = df.drop(columns=const_cols)

df['潜在可预防VTE'] = df['潜在可预防VTE'].map({True: 1, False: 0, 'True': 1, 'False': 0})

missing_col = '首次VTE中高风险评分日期与机械预防日期差值'
if missing_col in df.columns:
    df[missing_col + '_无'] = df[missing_col].isnull().astype(int)
    df[missing_col] = df[missing_col].fillna(0)

df['入院日期_dt'] = pd.to_datetime(df['入院日期'])
cutoff = pd.to_datetime('2025-03-31')

train_df = df[df['入院日期_dt'] <= cutoff].copy()
test_df = df[df['入院日期_dt'] > cutoff].copy()

cols_to_drop = ['入院日期', '入院日期_dt']
train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

train_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\train_data.csv', index=False)
test_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\test_data.csv', index=False)

outcome = '潜在可预防VTE'
all_cols = [c for c in train_df.columns if c != outcome]

continuous_vars = []
categorical_vars = []

for col in all_cols:
    unique_vals = train_df[col].dropna().unique()
    if len(unique_vals) > 10 and train_df[col].dtype in ['float64', 'int64']:
        continuous_vars.append(col)
    elif len(unique_vals) <= 10 or set(unique_vals) <= {0, 1}:
        categorical_vars.append(col)
    else:
        continuous_vars.append(col)

var_info = pd.DataFrame({
    'variable': continuous_vars + categorical_vars,
    'type': ['continuous'] * len(continuous_vars) + ['categorical'] * len(categorical_vars)
})
var_info.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\variable_info.csv', index=False)
