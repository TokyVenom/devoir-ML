#!/usr/bin/env python3
"""
models/train_advanced.py

Compare et entraîne plusieurs modèles pour les cibles x_wins et is_draw.
- RandomForest (class_weight)
- XGBoost (si installé) avec scale_pos_weight
- MLPClassifier (simple)
- 5-fold Stratified CV pour scoring
Sauvegarde les meilleurs modèles (par F1 macro) dans models/
"""
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / "ressources" / "dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Charger dataset
df = pd.read_csv(CSV_PATH)
features = [f"c{i}_x" for i in range(9)] + [f"c{i}_o" for i in range(9)]
X = df[features].values

# Détecter XGBoost
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return f1, classification_report(y_test, y_pred, zero_division=0), confusion_matrix(y_test, y_pred)

def run_for_target(y, target_name):
    print(f"\n=== Training for target: {target_name} ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None
    )

    results = []

    # 1) LogisticRegression baseline (class_weight balanced)
    lr = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
    f1_lr, report_lr, cm_lr = evaluate_model(lr, X_train, X_test, y_train, y_test)
    results.append(("LogisticRegression_balanced", lr, f1_lr, report_lr, cm_lr))
    print("LogReg balanced F1:", f1_lr)

    # 2) RandomForest (class_weight balanced)
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    f1_rf, report_rf, cm_rf = evaluate_model(rf, X_train, X_test, y_train, y_test)
    results.append(("RandomForest_balanced", rf, f1_rf, report_rf, cm_rf))
    print("RandomForest F1:", f1_rf)

    # 3) XGBoost (if available) with scale_pos_weight for imbalance
    if XGBOOST_AVAILABLE:
        # compute scale_pos_weight = n_neg / n_pos for binary
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) == 2:
            # assume class 1 is positive
            n_pos = counts[unique.tolist().index(1)] if 1 in unique else 0
            n_neg = len(y_train) - n_pos
            scale_pos_weight = max(1.0, n_neg / max(1, n_pos))
        else:
            scale_pos_weight = 1.0
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1)
        f1_xgb, report_xgb, cm_xgb = evaluate_model(xgb_clf, X_train, X_test, y_train, y_test)
        results.append(("XGBoost", xgb_clf, f1_xgb, report_xgb, cm_xgb))
        print("XGBoost F1:", f1_xgb)
    else:
        print("XGBoost not available, skipped.")

    # 4) MLPClassifier (simple)
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=RANDOM_STATE)
    f1_mlp, report_mlp, cm_mlp = evaluate_model(mlp, X_train, X_test, y_train, y_test)
    results.append(("MLP", mlp, f1_mlp, report_mlp, cm_mlp))
    print("MLP F1:", f1_mlp)

    # Sélection du meilleur modèle par F1 macro
    best = max(results, key=lambda r: r[2])
    name, model_obj, best_f1, best_report, best_cm = best
    print(f"\nBest model for {target_name}: {name} (F1={best_f1:.4f})")
    print("Classification report:\n", best_report)
    print("Confusion matrix:\n", best_cm)

    # Sauvegarde
    save_path = MODEL_DIR / f"{target_name}_{name}.pkl"
    joblib.dump(model_obj, save_path)
    print("Saved best model to:", save_path)

    # Cross-validated score (stratified)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model_obj, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    print("5-fold CV f1_macro scores:", cv_scores)
    print("CV mean f1_macro:", cv_scores.mean())

    return best

def main():
    # targets
    y_x = df['x_wins'].values
    y_draw = df['is_draw'].values

    best_x = run_for_target(y_x, "x_wins")
    best_draw = run_for_target(y_draw, "is_draw")

if __name__ == "__main__":
    main()
