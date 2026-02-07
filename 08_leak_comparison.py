# -*- coding: utf-8 -*-
"""
08_leak_comparison.py
对比分析：排除 vs 保留 预防相关变量

用户提出的5个争议变量：
- 首次VTE中高风险评分日期与机械预防日期差值
- 首次VTE中高风险评分日期与药物预防日期差值
- 规范预防
- 是否机械预防
- 是否药物预防

这些变量参与了"潜在可预防VTE"的定义，但用户认为它们也是判定目标变量的核心特征。
本脚本做两组对比：方案A（排除）vs 方案B（保留），输出对比CSV。
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.base import clone

BASE = r'C:\Users\Administrator\Desktop\2.5.388'
OUTCOME = '潜在可预防VTE'

# 读取已预处理的数据
train = pd.read_csv(os.path.join(BASE, 'train_data.csv'))
test = pd.read_csv(os.path.join(BASE, 'test_data.csv'))
ext = pd.read_csv(os.path.join(BASE, 'external_validation_data.csv'))

# 5个争议变量
dispute_vars = [
    '首次VTE中高风险评分日期与机械预防日期差值',
    '首次VTE中高风险评分日期与药物预防日期差值',
    '规范预防',
    '是否机械预防',
    '是否药物预防',
]

# 全部泄漏关键词（当前方案A的排除逻辑）
leak_keywords_full = ['预防', '医院相关性VTE', '我院相关VTE']

# 方案B的排除关键词（排除定义性变量，但保留争议变量）
leak_keywords_partial = ['医院相关性VTE', '我院相关VTE']
# 方案B中仍需排除的预防变量（哑变量展开等，但保留5个精确变量）
partial_exclude_patterns = ['预防措施_', '机械预防措施（VTE确诊前）_', '预防措施_新_']

def prepare_data(train_df, test_df, ext_df, mode='full_exclude'):
    """准备数据，mode='full_exclude'排除全部泄漏，mode='partial_keep'保留5个争议变量"""
    dfs = [train_df.copy(), test_df.copy(), ext_df.copy()]
    
    for d in dfs:
        # 删除非数值列
        for c in ['入院日期', 'dataset']:
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
        
        if mode == 'full_exclude':
            # 方案A：关键词全排除
            cols_drop = [c for c in d.columns if c != OUTCOME and any(kw in c for kw in leak_keywords_full)]
            d.drop(columns=cols_drop, inplace=True, errors='ignore')
        elif mode == 'partial_keep':
            # 方案B：保留5个争议变量，排除其他泄漏变量
            cols_drop = []
            for c in d.columns:
                if c == OUTCOME or c in dispute_vars:
                    continue
                # 排除定义性变量
                if any(kw in c for kw in leak_keywords_partial):
                    cols_drop.append(c)
                    continue
                # 排除预防措施哑变量（但不是争议变量）
                for pat in partial_exclude_patterns:
                    if c.startswith(pat):
                        cols_drop.append(c)
                        break
            d.drop(columns=cols_drop, inplace=True, errors='ignore')
    
    # 只保留数值列
    for i in range(3):
        num_cols = [c for c in dfs[i].columns if dfs[i][c].dtype in ['int64','float64','int32','float32','bool']]
        dfs[i] = dfs[i][num_cols]
    
    # 确保列一致
    common = sorted(set(dfs[0].columns) & set(dfs[1].columns) & set(dfs[2].columns))
    for i in range(3):
        dfs[i] = dfs[i][common]
    
    # 删除常量列
    const_cols = [c for c in dfs[0].columns if c != OUTCOME and dfs[0][c].nunique() <= 1]
    for i in range(3):
        dfs[i] = dfs[i].drop(columns=const_cols, errors='ignore')
    
    X_train = dfs[0].drop(columns=[OUTCOME]).fillna(0)
    y_train = dfs[0][OUTCOME].astype(int)
    X_test = dfs[1].drop(columns=[OUTCOME]).fillna(0)
    y_test = dfs[1][OUTCOME].astype(int)
    X_ext = dfs[2].drop(columns=[OUTCOME]).fillna(0)
    y_ext = dfs[2][OUTCOME].astype(int)
    
    return X_train, y_train, X_test, y_test, X_ext, y_ext

def evaluate(X_train, y_train, X_test, y_test, X_ext, y_ext, label):
    """训练6种模型，返回性能对比"""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.9, colsample_bytree=1.0, min_child_weight=3, reg_alpha=0.01, reg_lambda=0.5, scale_pos_weight=n_neg/n_pos if n_pos > 0 else 1, eval_metric='logloss', random_state=42, use_label_encoder=False),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    }
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_ext_s = scaler.transform(X_ext)
    
    results = []
    for name, model in models.items():
        use_scaled = name in ['SVM', 'Naive Bayes', 'KNN']
        Xtr = X_train_s if use_scaled else X_train.values
        Xte = X_test_s if use_scaled else X_test.values
        Xex = X_ext_s if use_scaled else X_ext.values
        
        # 10折CV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_probs = cross_val_predict(clone(model), Xtr, y_train, cv=cv, method='predict_proba')[:, 1]
        cv_auc = roc_auc_score(y_train, cv_probs)
        
        # 全量训练
        model.fit(Xtr, y_train)
        test_auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
        ext_auc = roc_auc_score(y_ext, model.predict_proba(Xex)[:, 1])
        
        results.append({
            '方案': label, '模型': name, '特征数': X_train.shape[1],
            '训练集AUC(10折CV)': round(cv_auc, 4),
            '测试集AUC': round(test_auc, 4),
            '外部验证AUC': round(ext_auc, 4),
        })
    return results

# ===== 方案A：排除全部泄漏变量（当前做法）=====
print("方案A：排除全部预防相关变量...")
Xa_tr, ya_tr, Xa_te, ya_te, Xa_ex, ya_ex = prepare_data(train, test, ext, 'full_exclude')
print(f"  特征数: {Xa_tr.shape[1]}")
results_a = evaluate(Xa_tr, ya_tr, Xa_te, ya_te, Xa_ex, ya_ex, 'A-排除全部')

# ===== 方案B：保留5个争议变量 =====
print("\n方案B：保留5个争议变量（规范预防、是否机械/药物预防、预防日期差值）...")
Xb_tr, yb_tr, Xb_te, yb_te, Xb_ex, yb_ex = prepare_data(train, test, ext, 'partial_keep')
print(f"  特征数: {Xb_tr.shape[1]}")
# 显示保留的争议变量
kept = [c for c in Xb_tr.columns if c in dispute_vars]
print(f"  保留的争议变量: {kept}")
results_b = evaluate(Xb_tr, yb_tr, Xb_te, yb_te, Xb_ex, yb_ex, 'B-保留争议变量')

# ===== 输出对比 =====
all_results = pd.DataFrame(results_a + results_b)
all_results.to_csv(os.path.join(BASE, 'leak_comparison_results.csv'), index=False, encoding='utf-8-sig')

# 透视表
print("\n" + "=" * 80)
print("对比结果")
print("=" * 80)
for metric in ['训练集AUC(10折CV)', '测试集AUC', '外部验证AUC']:
    pivot = all_results.pivot_table(index='模型', columns='方案', values=metric)
    pivot['差值(B-A)'] = pivot.iloc[:, 1] - pivot.iloc[:, 0] if pivot.shape[1] >= 2 else 0
    print(f"\n{metric}:")
    print(pivot.to_string())

print(f"\n对比结果已保存: leak_comparison_results.csv")
print(f"\n结论说明:")
print(f"方案A（排除全部）: {Xa_tr.shape[1]}个特征 - 排除所有预防相关变量，最保守")
print(f"方案B（保留争议）: {Xb_tr.shape[1]}个特征 - 保留规范预防、是否机械/药物预防等5个变量")
print(f"如果方案B的AUC显著高于方案A，说明这些变量确实提供了额外预测信息（而非仅仅是定义性泄漏）")
