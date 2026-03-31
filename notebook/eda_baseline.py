import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

# --- Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(BASE_DIR, 'ressources', 'dataset.csv')
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

# --- Chargement
print("[EDA] Chargement du dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("[EDA] shape:", df.shape)
print(df[['x_wins', 'is_draw']].value_counts(dropna=False))

# --- Colonnes features
features = []
for i in range(9):
    features.append(f"c{i}_x")
    features.append(f"c{i}_o")

# --- Aperçu rapide
print("\n[EDA] Aperçu (head):")
print(df.head())

# --- Distribution des cibles
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.countplot(x='x_wins', data=df)
plt.title('Distribution x_wins')
plt.subplot(1,2,2)
sns.countplot(x='is_draw', data=df)
plt.title('Distribution is_draw')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "distributions.png"))
print("[EDA] Distributions sauvegardées:", os.path.join(MODEL_DIR, "distributions.png"))

# --- Occupation par case quand X gagne
x_wins_df = df[df['x_wins'] == 1]
occ_x_when_win = [int(x_wins_df[f'c{i}_x'].sum()) for i in range(9)]
plt.figure(figsize=(4,4))
sns.heatmap(np.array(occ_x_when_win).reshape(3,3), annot=True, fmt='d', cmap='Blues')
plt.title('Occurences de X par case (positions où X gagne)')
plt.savefig(os.path.join(MODEL_DIR, "occ_x_when_win.png"))
print("[EDA] Heatmap occupation X when win saved:", os.path.join(MODEL_DIR, "occ_x_when_win.png"))

# --- Corrélation features vs cibles
corr = df[features + ['x_wins', 'is_draw']].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Heatmap corrélations features vs targets')
plt.savefig(os.path.join(MODEL_DIR, "corr_heatmap.png"))
print("[EDA] Correlation heatmap saved:", os.path.join(MODEL_DIR, "corr_heatmap.png"))

# --- Préparation X, y
X = df[features].values
y_x = df['x_wins'].values
y_draw = df['is_draw'].values

# --- Fonction d'entraînement et d'évaluation
def train_eval_logreg(X, y, target_name, save_path):
    print(f"\n[TRAIN] {target_name} - split train/test")
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[TRAIN] {target_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"[TRAIN] {target_name} - Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, zero_division=0))
    joblib.dump(model, save_path)
    print(f"[TRAIN] Model saved to {save_path}")
    return model

# --- Entraînement x_wins
model_x = train_eval_logreg(X, y_x, 'x_wins', os.path.join(MODEL_DIR, 'logreg_x_wins.pkl'))

# --- Entraînement is_draw
model_draw = train_eval_logreg(X, y_draw, 'is_draw', os.path.join(MODEL_DIR, 'logreg_is_draw.pkl'))

# --- Visualiser coefficients mappés sur plateau
def plot_coeffs(model, title_prefix, out_png):
    coefs = model.coef_.ravel()
    cx = coefs[0::2]
    co = coefs[1::2]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    sns.heatmap(np.array(cx).reshape(3,3), annot=True, cmap='RdBu', center=0)
    plt.title(f'{title_prefix} - coeffs c_i_x')
    plt.subplot(1,2,2)
    sns.heatmap(np.array(co).reshape(3,3), annot=True, cmap='RdBu', center=0)
    plt.title(f'{title_prefix} - coeffs c_i_o')
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[PLOT] Coeffs saved to {out_png}")

plot_coeffs(model_x, 'x_wins', os.path.join(MODEL_DIR, 'coeffs_x_wins.png'))
plot_coeffs(model_draw, 'is_draw', os.path.join(MODEL_DIR, 'coeffs_is_draw.png'))

print("\n[FIN] EDA et baselines terminés. Fichiers modèles et figures dans:", MODEL_DIR)
