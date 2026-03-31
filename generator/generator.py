import itertools
import os
import csv
from functools import lru_cache
from typing import List, Tuple, Optional

# Representation
# board: list[int] length 9 with values: 0 empty, 1 X, 2 O

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def winner(board: Tuple[int, ...]) -> Optional[int]:
    """
    Retourne 1 si X a gagné, 2 si O a gagné, None sinon.
    """
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_terminal(board: Tuple[int, ...]) -> bool:
    """
    True si la position est terminale (victoire ou plateau plein).
    """
    if winner(board) is not None:
        return True
    return all(cell != 0 for cell in board)


def valid_state(board: Tuple[int, ...]) -> bool:
    """
    Vérifie que l'état est valide:
    - pas de case occupée par X et O simultanément (impossible ici),
    - différence du nombre de X et O est 0 ou 1,
    - pas de deux gagnants simultanés,
    - si un gagnant existe, la configuration doit être cohérente avec le nombre de coups.
    """
    nx = sum(1 for v in board if v == 1)
    no = sum(1 for v in board if v == 2)
    # On autorise seulement états où c'est au tour de X => nx == no
    if nx != no:
        return False

    w = winner(board)
    # Si deux gagnants (impossible avec winner implémenté), reject
    # Vérifier cohérence: si gagnant X, alors nx must be no+1 normally,
    # mais ici nx==no so on exclut positions déjà terminales.
    if w is not None:
        # Exclure positions terminales (on veut états où c'est au tour de X et non-terminal)
        return False

    return True


# --- Minimax with alpha-beta and memoization ---
# Outcome values: 1 => X wins, 0 => draw, -1 => O wins
def board_to_key(board: Tuple[int, ...]) -> Tuple[int, ...]:
    return board


@lru_cache(maxsize=None)
def minimax_outcome(board: Tuple[int, ...]) -> int:
    """
    Retourne l'outcome optimal depuis la position donnée, en supposant que
    le joueur courant est déterminé par le nombre de X et O:
      - si nx == no => c'est au tour de X
      - sinon => au tour de O
    Valeurs retournées: 1 (X gagne), 0 (draw), -1 (O gagne)
    Utilise alpha-beta via paramètres internes (implémentation classique).
    """
    # Terminal ?
    w = winner(board)
    if w == 1:
        return 1
    if w == 2:
        return -1
    if all(cell != 0 for cell in board):
        return 0  # draw

    nx = sum(1 for v in board if v == 1)
    no = sum(1 for v in board if v == 2)
    turn = 1 if nx == no else 2  # 1 -> X, 2 -> O

    # For alpha-beta we implement a simple negamax variant with caching.
    # But since we use lru_cache on minimax_outcome, we implement a helper.
    def negamax(node_board: Tuple[int, ...], player: int) -> int:
        """
        Returns score from perspective of X:
        If player == 1 (X to move), we try to maximize outcome.
        If player == 2 (O to move), we try to minimize outcome.
        """
        w_local = winner(node_board)
        if w_local == 1:
            return 1
        if w_local == 2:
            return -1
        if all(cell != 0 for cell in node_board):
            return 0

        # X to move
        if player == 1:
            best = -2  # worse than -1
            for i in range(9):
                if node_board[i] == 0:
                    nb = list(node_board)
                    nb[i] = 1
                    nbt = tuple(nb)
                    val = minimax_outcome(nbt)
                    if val > best:
                        best = val
                        if best == 1:
                            break  # alpha-beta pruning: best possible
            return best
        else:
            # O to move: minimize X's outcome
            best = 2
            for i in range(9):
                if node_board[i] == 0:
                    nb = list(node_board)
                    nb[i] = 2
                    nbt = tuple(nb)
                    val = minimax_outcome(nbt)
                    if val < best:
                        best = val
                        if best == -1:
                            break
            return best

    return negamax(board, turn)


def encode_row(board: Tuple[int, ...]) -> List[int]:
    """
    Encode en 18 features: c0_x,c0_o,c1_x,c1_o,...,c8_x,c8_o
    """
    row = []
    for v in board:
        row.append(1 if v == 1 else 0)
        row.append(1 if v == 2 else 0)
    return row


def generate_dataset(csv_path: str, verbose: bool = True) -> int:
    """
    Parcourt tous les états (3^9) et sélectionne ceux valides où c'est au tour de X
    (non-terminal). Pour chaque état, calcule minimax_outcome et écrit une ligne CSV.
    Retourne le nombre de lignes écrites.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = [f"c{i}_x" for i in range(9)] + [f"c{i}_o" for i in range(9)]
    headers = [val for pair in zip([f"c{i}_x" for i in range(9)], [f"c{i}_o" for i in range(9)]) for val in pair]
    headers += ["x_wins", "is_draw"]

    total = 0
    written = 0

    # iterate over all 3^9 boards
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for digits in itertools.product((0, 1, 2), repeat=9):
            total += 1
            board = tuple(digits)
            if not valid_state(board):
                continue
            # compute outcome via minimax
            outcome = minimax_outcome(board)  # 1,0,-1
            x_wins = 1 if outcome == 1 else 0
            is_draw = 1 if outcome == 0 else 0
            row = encode_row(board) + [x_wins, is_draw]
            writer.writerow(row)
            written += 1
            if verbose and written % 1000 == 0:
                print(f"[generator] {written} states written...")

    if verbose:
        print(f"[generator] scanned {total} total boards, wrote {written} valid X-turn states to {csv_path}")
    return written


# --- Tests unitaires simples ---
def run_unit_tests():
    """
    Tests simples pour valider Minimax sur quelques positions connues.
    Board indexing:
    0 1 2
    3 4 5
    6 7 8
    """
    tests = []

    # Test 1: X can win immediately (X to move)
    # X at 0 and 1, empty 2 -> X plays 2 and wins
    b1 = (1, 1, 0,
          2, 2, 0,
          0, 0, 0)
    # fill with zeros to length 9
    b1 = tuple(b1)
    tests.append((b1, 1, "X immediate win"))

    # Test 2: Forced draw position (example symmetric)
    # Empty board with some moves leading to draw with perfect play is tricky;
    # we use a small constructed position where perfect play leads to draw.
    # Example: X at center, O at corner, X at opposite corner -> still complex.
    b2 = (1, 0, 0,
          0, 2, 0,
          0, 0, 0)
    b2 = tuple(b2)
    # outcome unknown a priori; we won't assert exact, just ensure function runs
    tests.append((b2, None, "non-trivial position (no assertion)"))

    # Test 3: O already winning (should be excluded by valid_state but test minimax)
    b3 = (2, 2, 2,
          0, 0, 0,
          0, 0, 0)
    b3 = tuple(b3)
    tests.append((b3, -1, "O already won"))

    print("[tests] Running unit tests...")
    for board, expected, desc in tests:
        out = minimax_outcome(board)
        print(f"[tests] {desc}: outcome={out} (expected {expected})")
        if expected is not None and out != expected:
            print(f"[tests] WARNING: test '{desc}' expected {expected} but got {out}")


def main():
    csv_path = os.path.join("../ressources", "dataset.csv")
    print("[generator] Starting dataset generation...")
    written = generate_dataset(csv_path)
    print(f"[generator] Done. {written} rows written to {csv_path}")
    # run quick validation checks
    print("[generator] Running quick validation checks...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    # checks
    cols = list(df.columns)
    print("[generator] Columns:", cols[:6], "...", cols[-2:])
    print("[generator] Number of rows:", len(df))
    # basic consistency checks
    bad_same_cell = 0
    for i in range(9):
        bad_same_cell += ((df[f"c{i}_x"] == 1) & (df[f"c{i}_o"] == 1)).sum()
    print("[generator] rows with same cell occupied by X and O:", int(bad_same_cell))
    nx = df[[f"c{i}_x" for i in range(9)]].sum(axis=1)
    no = df[[f"c{i}_o" for i in range(9)]].sum(axis=1)
    print("[generator] all rows have nb X == nb O ?", bool((nx == no).all()))
    inconsistent_labels = ((df["x_wins"] == 1) & (df["is_draw"] == 1)).sum()
    print("[generator] rows with x_wins=1 and is_draw=1 simultaneously:", int(inconsistent_labels))

    print("[generator] Running unit tests for minimax (sanity)...")
    run_unit_tests()
    print("[generator] Finished.")


if __name__ == "__main__":
    main()
