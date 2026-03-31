# interfaces/eval_models.py
from pathlib import Path
import joblib
import numpy as np

# Charger modèles calibrés
BASE_DIR = Path(__file__).resolve().parents[1]
CAL_DIR = BASE_DIR / "models" / "calibrated"

MODEL_X_PATH = CAL_DIR / "x_wins_XGBoost_calibrated.pkl"
MODEL_DRAW_PATH = CAL_DIR / "is_draw_MLP_calibrated.pkl"

if not MODEL_X_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_X_PATH}")
if not MODEL_DRAW_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_DRAW_PATH}")

model_x = joblib.load(str(MODEL_X_PATH))
model_draw = joblib.load(str(MODEL_DRAW_PATH))


def board_to_features(board):
    """
    board: list/tuple length 9 with values 0 empty, 1 X, 2 O
    returns: 18-features numpy array [c0_x,c0_o,...,c8_x,c8_o]
    """
    feats = []
    for v in board:
        feats.append(1 if v == 1 else 0)
        feats.append(1 if v == 2 else 0)
    return np.array(feats, dtype=int)


def evaluate_board_probs(board):
    """
    Retourne (p_x_wins, p_draw, p_o_wins_est)
    p_o_wins_est = 1 - p_x_wins - p_draw (approximation)
    """
    x = board_to_features(board).reshape(1, -1)
    px = float(model_x.predict_proba(x)[0][1])
    pd = float(model_draw.predict_proba(x)[0][1])
    po = max(0.0, 1.0 - px - pd)
    return px, pd, po


def choose_move_ml(board, legal_moves, alpha=0.5):
    """
    board: current board tuple/list
    legal_moves: list of indices (0..8) empty
    alpha: poids du draw dans le score final
    Retourne best_move, details
    details: list de tuples (move, p_x, p_draw, p_o_est, score)
    """
    best = None
    best_score = -1.0
    details = []
    for m in legal_moves:
        nb = list(board)
        nb[m] = 1  # X joue ce coup
        px, pd, po = evaluate_board_probs(tuple(nb))
        score = px + alpha * pd
        details.append((m, px, pd, po, score))
        if score > best_score:
            best_score = score
            best = m
    return best, details


def choose_move_hybrid(board, legal_moves, minimax_func, depth=3):
    """
    Hybrid: run minimax to depth, use ML eval at leaves.
    minimax_func should accept (board, depth, eval_fn) and return best move.
    eval_fn(board) must return a scalar score (higher = better for X).
    """
    def ml_eval_fn(b):
        px, pd, po = evaluate_board_probs(b)
        return px + 0.5 * pd - po  # coefficients ajustables

    return minimax_func(board, depth, ml_eval_fn)
