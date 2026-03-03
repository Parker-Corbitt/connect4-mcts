import json
import os
import random
import re
from urllib import error, request
import numpy as np

from .mcts import Node
from .connect4 import *

def query_ollama(state) -> list:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    legal_moves = []

    if isinstance(state, np.ndarray):
        legal_moves = valid_move(state)
        player = get_player_to_play(state)
        board_repr = state.tolist()
        user_prompt = (
            "Connect4 position:\n"
            f"board={board_repr}\n"
            f"player_to_move={player}\n"
            f"legal_moves={legal_moves}\n\n"
            "Return only JSON with your best moves from legal_moves and their probabilities, best first. "
            "Use this exact format: {\"moves\": [3 {.75}, 2 {.65}, 4 {.10}]}."
        )
    else:
        user_prompt = (
            "Given this Connect4 position, return best move columns as JSON.\n"
            f"position={state}\n\n"
            "Use this exact format: {\"moves\": [3 {.75}, 2 {.65}, 4 {.10}]}."

        )

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Connect4 move-ordering helper for MCTS. "
                    "Return only valid JSON, no prose."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0},
    }

    try:
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")

        response_obj = json.loads(raw)
        content = response_obj.get("message", {}).get("content", "{}")
        parsed_content = json.loads(content) if isinstance(content, str) else content

        if isinstance(parsed_content, dict):
            moves = parsed_content.get("moves", [])
        elif isinstance(parsed_content, list):
            moves = parsed_content
        else:
            moves = []

    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    normalized = []
    for move in moves:
        if isinstance(move, int):
            normalized.append(move)
            continue
        if isinstance(move, str) and re.fullmatch(r"\d+", move):
            normalized.append(int(move))

    if legal_moves:
        legal_set = set(legal_moves)
        normalized = [m for m in normalized if m in legal_set]

    # Keep order, remove duplicates.
    deduped = []
    seen = set()
    for move in normalized:
        if move not in seen:
            deduped.append(move)
            seen.add(move)
    return deduped

def random_play(grid):
    """
    Play a random game starting by state and player
    Return winner
    """

    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        selected_move = random.choice(moves)
        player_to_play = get_player_to_play(grid)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play

def random_play_improved(grid):

    def get_winning_moves(grid, moves, player):
        return [move for move in moves if play(grid, move, player=player)[1]]

    # If can win, win
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        player_to_play = get_player_to_play(grid)

        winning_moves = get_winning_moves(grid, moves, player_to_play)
        loosing_moves = get_winning_moves(grid, moves, -player_to_play)

        if len(winning_moves) > 0:
            selected_move = winning_moves[0]
        elif len(loosing_moves) == 1:
            selected_move = loosing_moves[0]
        else:
            selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play


def train_mcts_during(mcts, training_time):
    import time
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts)
        current = int(round(time.time() * 1000))
    return mcts

def train_mcts_once(mcts=None):

    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)

    node = mcts

    # selection
    while node.children is not None:
        # Select highest uct
        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # expansion, no expansion if terminal node
    moves = valid_move(node.state)
    if len(moves) > 0:

        if node.winner == 0:
            
            ollama_moves = query_ollama(node.state)
            if ollama_moves:
                print(ollama_moves)
            
            states = [(play(node.state, move), move) for move in moves]
            node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # simulation
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner

        # backpropagation
        parent = node
        while parent is not None:
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
            parent = parent.parent


    else:
        print('no valid moves, expended all')

    return mcts
