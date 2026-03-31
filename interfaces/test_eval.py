# interface/test_eval.py
from eval_models import board_to_features, evaluate_board_probs, choose_move_ml
import random

# exemples de plateaux (0 empty, 1 X, 2 O)
empty = (0,0,0, 0,0,0, 0,0,0)
midgame = (1,2,0, 0,1,0, 2,0,0)  # X au centre et coin, O en face
near_win = (1,1,0, 2,2,0, 0,0,0)  # X peut gagner en pos 2

def legal_moves(board):
    return [i for i,v in enumerate(board) if v==0]

def pretty(board):
    s = ""
    for i,v in enumerate(board):
        ch = "." if v==0 else ("X" if v==1 else "O")
        s += ch + ("\n" if i%3==2 else " ")
    return s

for b in [empty, midgame, near_win]:
    print("Board:\n", pretty(b))
    feats = board_to_features(b)
    print("Features sum (X count, O count):", sum(feats[0::2]), sum(feats[1::2]))
    px, pd, po = evaluate_board_probs(b)
    print(f"P(x_wins)={px:.3f}, P(draw)={pd:.3f}, est P(o_wins)={po:.3f}")
    moves = legal_moves(b)
    best, details = choose_move_ml(b, moves)
    print("Best ML move:", best)
    for d in details:
        print(" move", d[0], "px", f"{d[1]:.3f}", "pd", f"{d[2]:.3f}", "score", f"{d[4]:.3f}")
    print("-"*40)
