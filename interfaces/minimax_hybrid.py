# interfaces/minimax_hybrid.py
from typing import List, Optional, Tuple

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),  # rows
    (0,3,6),(1,4,7),(2,5,8),  # cols
    (0,4,8),(2,4,6)           # diags
]

def winner(board: Tuple[int, ...]) -> Optional[int]:
    """
    Retourne 1 si X gagne, 2 si O gagne, 0 si match nul, None si non terminal.
    board: tuple/list de longueur 9 avec valeurs 0/1/2
    """
    # vérifier victoire
    for a,b,c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    # si cases pleines -> draw
    if all(v != 0 for v in board):
        return 0
    return None

def _current_player(board: Tuple[int, ...]) -> int:
    """
    Détermine qui doit jouer: X (1) si nb X <= nb O, sinon O (2).
    """
    nx = sum(1 for v in board if v == 1)
    no = sum(1 for v in board if v == 2)
    return 1 if nx <= no else 2

def _legal_moves(board: Tuple[int, ...]) -> List[int]:
    return [i for i,v in enumerate(board) if v == 0]

def _minimax(board: Tuple[int, ...], depth: int, maximizing: bool) -> Tuple[int, Optional[int]]:
    """
    Minimax simple:
    - score +1 si X gagne, -1 si O gagne, 0 draw or depth limit
    - retourne (score, best_move)
    """
    w = winner(board)
    if w is not None:
        if w == 1:
            return 1, None
        if w == 2:
            return -1, None
        return 0, None  # draw

    if depth == 0:
        return 0, None  # heuristique neutre (simple stub)

    moves = _legal_moves(board)
    best_move = None

    if maximizing:
        best_score = -999
        for m in moves:
            nb = list(board)
            nb[m] = 1
            score, _ = _minimax(tuple(nb), depth-1, False)
            if score > best_score:
                best_score = score
                best_move = m
                if best_score == 1:
                    break
        return best_score, best_move
    else:
        best_score = 999
        for m in moves:
            nb = list(board)
            nb[m] = 2
            score, _ = _minimax(tuple(nb), depth-1, True)
            if score < best_score:
                best_score = score
                best_move = m
                if best_score == -1:
                    break
        return best_score, best_move

def choose_move_minimax_hybrid(board: Tuple[int, ...],
                               legal_moves: Optional[List[int]] = None,
                               depth: int = 3,
                               use_ml: bool = False) -> Optional[int]:
    """
    Interface simple pour choisir un coup via Minimax.
    - board: tuple/list longueur 9
    - legal_moves: liste d'indices vides (optionnel)
    - depth: profondeur de recherche
    - use_ml: param pour compatibilité (non utilisé dans ce stub)
    Retourne l'indice du coup choisi ou None.
    """
    if legal_moves is None:
        legal_moves = _legal_moves(board)
    if not legal_moves:
        return None

    player = _current_player(board)
    maximizing = True if player == 1 else False

    # Appel du minimax
    score, move = _minimax(tuple(board), depth, maximizing)
    if move is None:
        # fallback: choisir un coup légal aléatoire (prévisible)
        return legal_moves[0]
    return move
