# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\train_data.csv')
uni_results = pd.read_csv(r'C:\Users\Administrator\Desktop\2.5.388\python_univariate_results.csv')

outcome = '潜在可预防VTE'
sig_vars = uni_results[uni_results['P值'] < 0.1]['变量'].tolist()

var_mapping = {}
spss_vars = [outcome]
var_mapping[outcome] = 'Y_VTE'

for i, var in enumerate(sig_vars[:30]):
    short_name = f'X{i+1}'
    var_mapping[var] = short_name
    spss_vars.append(var)

spss_data = train_df[spss_vars].copy()
spss_data.columns = [var_mapping[c] for c in spss_data.columns]
spss_data.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\spss_data.csv', index=False)

mapping_df = pd.DataFrame([
    {'SPSS变量名': var_mapping[k], '原始变量名': k, 
     'P值': uni_results[uni_results['变量']==k]['P值'].values[0] if k in uni_results['变量'].values else None}
    for k in var_mapping.keys()
])
mapping_df.to_csv(r'C:\Users\Administrator\Desktop\2.5.388\spss_variable_mapping.csv', index=False, encoding='utf-8-sig')

spss_syntax = '''* SPSS分析语法文件.

GET DATA
  /TYPE=TXT
  /FILE="C:\\Users\\Administrator\\Desktop\\2.5.388\\spss_data.csv"
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
'''
for col in spss_data.columns:
    if spss_data[col].dtype == 'float64':
        spss_syntax += f'    {col} F8.4\n'
    else:
        spss_syntax += f'    {col} F8.0\n'

spss_syntax += '''.\nCACHE.\nEXECUTE.\n'''

continuous_spss = []
categorical_spss = []
for orig, spss in var_mapping.items():
    if orig == outcome:
        continue
    if train_df[orig].nunique() > 10:
        continuous_spss.append(spss)
    else:
        categorical_spss.append(spss)

if continuous_spss:
    spss_syntax += f'''
T-TEST GROUPS=Y_VTE(0 1)
  /VARIABLES={' '.join(continuous_spss[:10])}
  /CRITERIA=CI(.95).

NPAR TESTS
  /M-W={' '.join(continuous_spss[:10])} BY Y_VTE(0 1)
  /MISSING ANALYSIS.
'''

if categorical_spss:
    for var in categorical_spss[:15]:
        spss_syntax += f'''
CROSSTABS
  /TABLES={var} BY Y_VTE
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ
  /CELLS=COUNT ROW COLUMN.
'''

for spss_var in list(var_mapping.values())[1:]:
    spss_syntax += f'''
LOGISTIC REGRESSION VARIABLES Y_VTE
  /METHOD=ENTER {spss_var}
  /PRINT=CI(95)
  /CRITERIA=PIN(0.05) POUT(0.10) ITERATE(20) CUT(0.5).
'''

multi_vars = ['X5', 'X1', 'X14', 'X3', 'X10', 'X15', 'X12', 'X11', 'X2', 'X13']

spss_syntax += f'''
LOGISTIC REGRESSION VARIABLES Y_VTE
  /METHOD=ENTER {' '.join(multi_vars)}
  /PRINT=CI(95) GOODFIT
  /CRITERIA=PIN(0.05) POUT(0.10) ITERATE(20) CUT(0.5).

LOGISTIC REGRESSION VARIABLES Y_VTE
  /METHOD=FSTEP(LR) {' '.join(multi_vars)}
  /PRINT=CI(95) GOODFIT
  /CRITERIA=PIN(0.05) POUT(0.10) ITERATE(20) CUT(0.5).

LOGISTIC REGRESSION VARIABLES Y_VTE
  /METHOD=BSTEP(LR) {' '.join(multi_vars)}
  /PRINT=CI(95) GOODFIT
  /CRITERIA=PIN(0.05) POUT(0.10) ITERATE(20) CUT(0.5).
'''

with open(r'C:\Users\Administrator\Desktop\2.5.388\SPSS_analysis.sps', 'w', encoding='utf-8') as f:
    f.write(spss_syntax)
