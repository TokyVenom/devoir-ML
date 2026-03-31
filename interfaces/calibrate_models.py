# interfaces/calibrate_models.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "ressources" / "dataset.csv"
MODELS = BASE / "models"
OUT = MODELS / "calibrated"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
features = [f"c{i}_x" for i in range(9)] + [f"c{i}_o" for i in range(9)]
X = df[features].values

# Utiliser un split pour l'entraînement du calibrateur (optionnel)
# Ici on garde X_train/X_cal pour fit du calibrateur (cv interne fera les folds)
X_train, X_cal_all, y_train_dummy, y_cal_all = train_test_split(
    X, df['x_wins'].values, test_size=0.2, random_state=42, stratify=df['x_wins'].values
)

# ---------- Calibrer x_wins ----------
print("Calibrating x_wins...")
model_x = joblib.load(MODELS / "x_wins_XGBoost.pkl")

# CalibratedClassifierCV va effectuer ses propres CV fits (cv=5)
cal_x = CalibratedClassifierCV(estimator=model_x, method='sigmoid', cv=5)
# fit sur tout X (ou X_train) : il fera internal CV
cal_x.fit(X, df['x_wins'].values)
joblib.dump(cal_x, OUT / "x_wins_XGBoost_calibrated.pkl")
print("Calibrated x_wins saved to", OUT / "x_wins_XGBoost_calibrated.pkl")

# ---------- Calibrer is_draw ----------
print("Calibrating is_draw...")
model_d = joblib.load(MODELS / "is_draw_MLP.pkl")
cal_d = CalibratedClassifierCV(estimator=model_d, method='sigmoid', cv=5)
cal_d.fit(X, df['is_draw'].values)
joblib.dump(cal_d, OUT / "is_draw_MLP_calibrated.pkl")
print("Calibrated is_draw saved to", OUT / "is_draw_MLP_calibrated.pkl")

print("Calibration finished.")
