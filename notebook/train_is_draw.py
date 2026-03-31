#!/usr/bin/env python3
"""
notebook/train_is_draw.py

Script ciblé pour améliorer la prédiction de is_draw :
- compare LogisticRegression (class_weight), SMOTE + LogisticRegression, RandomForest (class_weight)
- affiche rapports et sauvegarde modèles
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib

BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / "ressources" / "dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)
features = [f"c{i}_x" for i in range(9)] + [f"c{i}_o" for i in range(9)]
X = df[features].values
y = df['is_draw'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

# 1) LogisticRegression with class_weight balanced
lr_bal = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
lr_bal.fit(X_train, y_train)
y_pred_lr = lr_bal.predict(X_test)
print("\nLogReg class_weight balanced")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print("Confusion:\n", confusion_matrix(y_test, y_pred_lr))
joblib.dump(lr_bal, MODEL_DIR / "logreg_is_draw_classweight.pkl")

# 2) SMOTE + LogisticRegression
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE resampled train distribution:", dict(zip(*np.unique(y_res, return_counts=True))))
lr_sm = LogisticRegression(max_iter=2000, solver='liblinear')
lr_sm.fit(X_res, y_res)
y_pred_sm = lr_sm.predict(X_test)
print("\nLogReg with SMOTE")
print(classification_report(y_test, y_pred_sm, zero_division=0))
print("Confusion:\n", confusion_matrix(y_test, y_pred_sm))
joblib.dump(lr_sm, MODEL_DIR / "logreg_is_draw_smote.pkl")

# 3) RandomForest with class_weight balanced
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandomForest class_weight balanced")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("Confusion:\n", confusion_matrix(y_test, y_pred_rf))
joblib.dump(rf, MODEL_DIR / "rf_is_draw.pkl")

# Compare F1 for class 1 (is_draw)
f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
f1_sm = f1_score(y_test, y_pred_sm, pos_label=1, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
print("\nF1 (is_draw=1) summary: LogReg_bal=%.4f, LogReg_SMOTE=%.4f, RF=%.4f" % (f1_lr, f1_sm, f1_rf))

print("\nSaved models in:", MODEL_DIR)
