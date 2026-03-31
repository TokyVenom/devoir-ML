# scripts/evaluate_models.py
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score

BASE = Path(__file__).resolve().parents[1]
CAL_DIR = BASE / "models" / "calibrated"

# chemins modèles (adapter si noms différents)
MODEL_X = CAL_DIR / "x_wins_XGBoost_calibrated.pkl"
MODEL_DRAW = CAL_DIR / "is_draw_MLP_calibrated.pkl"

# jeu de test
# On suppose un CSV test avec colonnes features 18 colonnes et labels x_wins,is_draw
TEST_CSV = BASE / "data" / "test_positions.csv"

def load_models():
    mx = joblib.load(MODEL_X)
    md = joblib.load(MODEL_DRAW)
    return mx, md

def load_test():
    df = pd.read_csv(TEST_CSV)
    # features attendues : x0,o0,x1,o1,...,x8,o8 (18 colonnes)
    X = df[[c for c in df.columns if c.startswith("x") or c.startswith("o")]].values
    y_x = df["x_wins"].values
    y_d = df["is_draw"].values
    return X, y_x, y_d, df

def eval_binary(model, X, y_true, pos_label=1):
    # probabilités pour la classe positive
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback : decision_function
        probs = model.decision_function(X)
    y_pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {"auc": auc, "ap": ap, "f1": f1, "precision": prec, "recall": rec, "confusion_matrix": cm, "probs": probs, "pred": y_pred}

def get_importances(model):
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        # si multiclass, prendre la première ligne ou la norme
        if coef.ndim > 1:
            coef = np.abs(coef).sum(axis=0)
        else:
            coef = coef.ravel()
        return coef
    return None

def main():
    print("Chargement modèles et données...")
    mx, md = load_models()
    X, y_x, y_d, df = load_test()
    print("Évaluation x_wins (XGBoost)...")
    res_x = eval_binary(mx, X, y_x)
    print("Évaluation is_draw (MLP)...")
    res_d = eval_binary(md, X, y_d)

    # importances
    imp_x = get_importances(mx)
    imp_d = get_importances(md)

    # afficher résumé
    def print_res(name, res, imp):
        print(f"\n--- Résultats pour {name} ---")
        print(f"AUC: {res['auc']:.4f}   AP: {res['ap']:.4f}")
        print(f"F1: {res['f1']:.4f}   Precision: {res['precision']:.4f}   Recall: {res['recall']:.4f}")
        print("Confusion matrix:\n", res["confusion_matrix"])
        if imp is not None:
            # features names
            names = []
            for i in range(9):
                names.append(f"x{i}")
                names.append(f"o{i}")
            idx = np.argsort(np.abs(imp))[::-1][:8]
            print("Top features (name, value):")
            for i in idx:
                print(names[i], float(imp[i]))
    print_res("x_wins", res_x, imp_x)
    print_res("is_draw", res_d, imp_d)

    # sauvegarder CSV des probabilités et prédictions
    out = df.copy()
    out["p_x_wins"] = res_x["probs"]
    out["pred_x_wins"] = res_x["pred"]
    out["p_is_draw"] = res_d["probs"]
    out["pred_is_draw"] = res_d["pred"]
    out.to_csv(BASE / "results" / "eval_predictions.csv", index=False)
    print("\nFichier results/eval_predictions.csv écrit.")

if __name__ == "__main__":
    main()
