# interface/streamlit_app.py
import streamlit as st
from interfaces.eval_models import evaluate_board_probs, choose_move_ml
from interfaces.minimax_hybrid import choose_move_minimax_hybrid, winner
import random

st.set_page_config(page_title="IA Morpion", layout="centered")
st.title("IA Morpion")

# Etat initial
if "board" not in st.session_state:
    st.session_state.board = [0] * 9
if "log" not in st.session_state:
    st.session_state.log = []
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "current_player" not in st.session_state:
    st.session_state.current_player = 1  # 1 = X, 2 = O
if "mode" not in st.session_state:
    st.session_state.mode = "ML"
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.5

def reset_board():
    st.session_state.board = [0] * 9
    st.session_state.log = []
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.current_player = 1

def end_if_terminal():
    w = winner(tuple(st.session_state.board))
    if w is not None:
        st.session_state.game_over = True
        st.session_state.winner = w
        return True
    return False

def place_move(index, player):
    st.session_state.board[index] = player
    st.session_state.log.append((player, index))

def ai_choose_and_place(alpha):
    legal = [i for i, v in enumerate(st.session_state.board) if v == 0]
    if not legal:
        return
    try:
        res = choose_move_ml(tuple(st.session_state.board), legal, alpha=alpha)
        if isinstance(res, tuple) and len(res) >= 1:
            move = res[0]
        else:
            move = res
    except Exception as e:
        st.session_state.log.append(("IA_error", str(e)))
        move = None
    try:
        move = int(move) if move is not None else None
    except Exception:
        move = None
    if move is None or move not in legal:
        move = random.choice(legal)
    place_move(move, 2)
    st.session_state.log.append(("IA", move))

def on_cell_click(i):
    # si partie terminée, ne rien faire
    if st.session_state.game_over:
        return

    mode = st.session_state.mode
    # PvP mode: alternate current_player and place that symbol
    if mode == "PvP":
        player = st.session_state.current_player
        if st.session_state.board[i] == 0:
            place_move(i, player)
            # vérifier fin de partie
            if end_if_terminal():
                return
            # alterner joueur
            st.session_state.current_player = 2 if player == 1 else 1
        return

    # Modes avec IA ou Minimax: l'utilisateur joue toujours X (1)
    if st.session_state.board[i] != 0:
        return
    place_move(i, 1)
    if end_if_terminal():
        return

    # IA joue O
    if mode == "ML":
        ai_choose_and_place(st.session_state.alpha)
    elif mode == "Minimax":
        legal = [j for j, v in enumerate(st.session_state.board) if v == 0]
        if legal:
            move = choose_move_minimax_hybrid(tuple(st.session_state.board), legal, depth=3)
            try:
                move = int(move)
            except Exception:
                move = None
            if move is None or move not in legal:
                move = random.choice(legal)
            place_move(move, 2)
            st.session_state.log.append(("IA_minimax", move))
    else:  # Hybrid
        legal = [j for j, v in enumerate(st.session_state.board) if v == 0]
        if legal:
            move = choose_move_minimax_hybrid(tuple(st.session_state.board), legal, depth=3, use_ml=True)
            try:
                move = int(move)
            except Exception:
                move = None
            if move is None or move not in legal:
                move = random.choice(legal)
            place_move(move, 2)
            st.session_state.log.append(("IA_hybrid", move))

    end_if_terminal()

# Controls
st.sidebar.markdown("### Réglages")
st.session_state.mode = st.sidebar.selectbox("Mode IA", ["ML", "Minimax", "Hybrid", "PvP"], index=["ML","Minimax","Hybrid","PvP"].index(st.session_state.mode))
st.session_state.alpha = st.sidebar.slider("Poids draw alpha", 0.0, 1.0, st.session_state.alpha, 0.05)
st.sidebar.button("Reset", on_click=reset_board)

# Plateau affichage avec callbacks
cols = st.columns(3)
for i in range(9):
    col = cols[i % 3]
    val = st.session_state.board[i]
    label = "." if val == 0 else ("X" if val == 1 else "O")
    # désactiver boutons si partie terminée
    disabled = st.session_state.game_over
    col.button(label, key=f"cell_{i}", on_click=on_cell_click, args=(i,), disabled=disabled)

# Texte plateau
st.markdown("### Plateau")
grid = ""
for i, v in enumerate(st.session_state.board):
    grid += (". " if v == 0 else ("X " if v == 1 else "O "))
    if i % 3 == 2:
        grid += "\n"
st.text(grid)

# Message de fin de partie
if st.session_state.game_over:
    w = st.session_state.winner
    if w == 1:
        st.success("Partie terminée — X a gagné")
    elif w == 2:
        st.success("Partie terminée — O a gagné")
    elif w == 0:
        st.info("Partie terminée — Match nul")
    st.markdown("Appuyez sur Reset pour recommencer")

# Probabilités et debug
st.markdown("### Probabilités pour la position courante")
try:
    px, pd, po = evaluate_board_probs(tuple(st.session_state.board))
    st.write(f"**P(x_wins)** = {px:.3f}   **P(draw)** = {pd:.3f}   **P(o_wins est.)** = {po:.3f}")
except Exception as e:
    st.write("Erreur evaluate_board_probs:", e)

st.markdown("### Debug")
st.write("Board:", st.session_state.board)
st.write("Current player:", st.session_state.current_player)
st.write("Game over:", st.session_state.game_over)
st.write("Winner:", st.session_state.winner)
st.markdown("### Log des derniers coups")
for entry in st.session_state.log[-20:]:
    st.write(entry)
