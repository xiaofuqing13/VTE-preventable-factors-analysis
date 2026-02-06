# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\train_data.csv')

outcome = '潜在可预防VTE'
y = train_df[outcome]

all_vars = [c for c in train_df.columns if c != outcome and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]

valid_vars = []
for var in all_vars:
    if train_df[var].std() > 0 and train_df[var].notna().sum() > 10:
        valid_vars.append(var)

univariate_results = []
for var in valid_vars:
    try:
        X = train_df[[var]].copy()
        X = sm.add_constant(X)
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        if len(y_clean) < 20:
            continue
        model = sm.Logit(y_clean, X_clean)
        result = model.fit(disp=0, maxiter=100)
        coef = result.params[var]
        se = result.bse[var]
        p_val = result.pvalues[var]
        OR = np.exp(coef)
        CI_low = np.exp(coef - 1.96 * se)
        CI_high = np.exp(coef + 1.96 * se)
        univariate_results.append({'变量': var, 'β系数': round(coef, 4), 'SE': round(se, 4),
            'OR': round(OR, 3), '95%CI下限': round(CI_low, 3), '95%CI上限': round(CI_high, 3), 'P值': round(p_val, 4)})
    except:
        continue

uni_df = pd.DataFrame(univariate_results)
uni_df = uni_df.sort_values('P值')
uni_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\python_univariate_results.csv', index=False, encoding='utf-8-sig')

sig_vars_01 = uni_df[uni_df['P值'] < 0.1]['变量'].tolist()
candidate_vars = sig_vars_01.copy()

if len(candidate_vars) > 1:
    X_cand = train_df[candidate_vars].copy()
    corr_matrix = X_cand.corr().abs()
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                var_i = corr_matrix.columns[i]
                var_j = corr_matrix.columns[j]
                p_i = uni_df[uni_df['变量'] == var_i]['P值'].values[0]
                p_j = uni_df[uni_df['变量'] == var_j]['P值'].values[0]
                if p_i > p_j:
                    to_remove.add(var_i)
                else:
                    to_remove.add(var_j)
    candidate_vars = [v for v in candidate_vars if v not in to_remove]

max_vars = min(len(candidate_vars), int(min(y.sum(), len(y) - y.sum()) / 10))
if max_vars < len(candidate_vars):
    candidate_vars = candidate_vars[:max_vars]

if len(candidate_vars) >= 1:
    X_multi = train_df[candidate_vars].copy()
    X_multi = sm.add_constant(X_multi)
    mask = X_multi.notna().all(axis=1) & y.notna()
    X_clean = X_multi[mask]
    y_clean = y[mask]
    try:
        model_multi = sm.Logit(y_clean, X_clean)
        result_multi = model_multi.fit(disp=0, maxiter=200)
        multi_results = []
        for var in candidate_vars:
            coef = result_multi.params[var]
            se = result_multi.bse[var]
            p_val = result_multi.pvalues[var]
            OR = np.exp(coef)
            CI_low = np.exp(coef - 1.96 * se)
            CI_high = np.exp(coef + 1.96 * se)
            multi_results.append({'变量': var, 'β系数': round(coef, 4), 'SE': round(se, 4),
                'OR': round(OR, 3), '95%CI下限': round(CI_low, 3), '95%CI上限': round(CI_high, 3), 'P值': round(p_val, 4)})
        multi_df = pd.DataFrame(multi_results)
        multi_df = multi_df.sort_values('P值')
        multi_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\python_multivariate_results.csv', index=False, encoding='utf-8-sig')
        with open(r'C:\Users\Administrator\Desktop\2.5.388\python_model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(result_multi.summary().as_text())
    except:
        pass
