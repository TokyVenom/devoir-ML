# scripts/generate_test_positions.py
from pathlib import Path
import csv

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)
OUT_CSV = OUT / "test_positions.csv"

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

def winner(board):
    for a,b,c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(v != 0 for v in board):
        return 0
    return None

def dfs(board, turn, rows):
    w = winner(board)
    if w is not None:
        feats = []
        for v in board:
            feats.append(1 if v == 1 else 0)
            feats.append(1 if v == 2 else 0)
        x_wins = 1 if w == 1 else 0
        is_draw = 1 if w == 0 else 0
        rows.append(feats + [x_wins, is_draw])
        return
    for i, v in enumerate(board):
        if v == 0:
            nb = list(board)
            nb[i] = turn
            dfs(tuple(nb), 2 if turn == 1 else 1, rows)

def main():
    rows = []
    dfs((0,0,0,0,0,0,0,0,0), 1, rows)
    header = []
    for i in range(9):
        header.append(f"x{i}")
        header.append(f"o{i}")
    header += ["x_wins", "is_draw"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("Wrote", OUT_CSV, "with", len(rows), "rows")

if __name__ == "__main__":
    main()
