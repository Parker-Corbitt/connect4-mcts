import json
import os
import random
import re
import time
from urllib import error, request

import numpy as np

from .mcts import Node
from .connect4 import *

_POLICY_CACHE = {}
_LLM_CALLS = 0
_LLM_CACHE_HITS = 0


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


LLM_POLICY_ENABLED = _env_bool("LLM_POLICY_ENABLED", True)
LLM_POLICY_MAX_CALLS = _env_int("LLM_POLICY_MAX_CALLS", 256)
LLM_POLICY_MAX_DEPTH = _env_int("LLM_POLICY_MAX_DEPTH", 6)
LLM_TIMEOUT_SECONDS = _env_float("OLLAMA_TIMEOUT_SECONDS", 10.0)
MCTS_C_PUCT = _env_float("MCTS_C_PUCT", 1.25)


def _center_policy(legal_moves):
    # Center columns are usually stronger in Connect4.
    raw = {move: float(4 - abs(3 - move)) for move in legal_moves}
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(legal_moves)
        return {move: uniform for move in legal_moves}
    return {move: raw[move] / total for move in legal_moves}


def _uniform_policy(legal_moves):
    if not legal_moves:
        return {}
    value = 1.0 / len(legal_moves)
    return {move: value for move in legal_moves}


def _normalize_policy(raw_policy, legal_moves):
    if not legal_moves:
        return {}

    center = _center_policy(legal_moves)
    if not raw_policy:
        return center

    scores = {}
    for move in legal_moves:
        base = max(0.0, float(raw_policy.get(move, 0.0)))
        # Keep every legal move explorable.
        scores[move] = base + (0.05 * center[move])

    total = sum(scores.values())
    if total <= 0:
        return center
    return {move: scores[move] / total for move in legal_moves}


def _parse_probability(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_move(move):
    if isinstance(move, int):
        return move
    if isinstance(move, str) and re.fullmatch(r"\d+", move):
        return int(move)
    return None


def _extract_move_scores(parsed_content, legal_moves):
    if isinstance(parsed_content, dict):
        moves = parsed_content.get("moves", [])
    elif isinstance(parsed_content, list):
        moves = parsed_content
    else:
        moves = []

    legal = set(legal_moves)
    ranked_moves = []
    raw_scores = {}

    for item in moves:
        move = None
        score = None

        if isinstance(item, dict):
            move = _parse_move(
                item.get("col", item.get("move", item.get("column")))
            )
            score = _parse_probability(
                item.get("p", item.get("prob", item.get("probability")))
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            move = _parse_move(item[0])
            if len(item) > 1:
                score = _parse_probability(item[1])
        else:
            move = _parse_move(item)

        if move is None or move not in legal:
            continue

        ranked_moves.append(move)
        if score is not None and score > 0:
            raw_scores[move] = max(raw_scores.get(move, 0.0), score)

    if raw_scores:
        return raw_scores

    # If the LLM returns ranking only, turn rank into soft scores.
    rank_scores = {}
    seen = set()
    for idx, move in enumerate(ranked_moves):
        if move in seen:
            continue
        seen.add(move)
        rank_scores[move] = 1.0 / (idx + 1)
    return rank_scores


def query_ollama(state, legal_moves):
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    player = get_player_to_play(state)
    board_repr = state.tolist()
    user_prompt = (
        "Connect4 position:\n"
        f"board={board_repr}\n"
        f"player_to_move={player}\n"
        f"legal_moves={legal_moves}\n\n"
        "Return JSON only with this schema:\n"
        "{\"moves\": [{\"col\": 3, \"p\": 0.55}, {\"col\": 2, \"p\": 0.25}]}\n"
        "Rules: include only legal columns; probabilities must be > 0 and sum close to 1."
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
        with request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")

        response_obj = json.loads(raw)
        content = response_obj.get("message", {}).get("content", "{}")
        parsed_content = json.loads(content) if isinstance(content, str) else content
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return {}
    return _extract_move_scores(parsed_content, legal_moves)


def get_llm_policy_stats():
    return {
        "llm_calls": _LLM_CALLS,
        "cache_entries": len(_POLICY_CACHE),
        "cache_hits": _LLM_CACHE_HITS,
    }


def _get_policy_for_node(node, legal_moves):
    global _LLM_CALLS
    global _LLM_CACHE_HITS

    state_key = f"llm:{to_state(node.state)}"
    cached = _POLICY_CACHE.get(state_key)
    if cached is not None:
        _LLM_CACHE_HITS += 1
        return cached

    can_query = (
        LLM_POLICY_ENABLED
        and node.depth <= LLM_POLICY_MAX_DEPTH
        and (LLM_POLICY_MAX_CALLS < 0 or _LLM_CALLS < LLM_POLICY_MAX_CALLS)
    )

    if can_query:
        _LLM_CALLS += 1
        raw_policy = query_ollama(node.state, legal_moves)
    else:
        raw_policy = {}

    policy = _normalize_policy(raw_policy, legal_moves)
    _POLICY_CACHE[state_key] = policy
    return policy


def _get_pure_policy_for_node(node, legal_moves):
    state_key = f"pure:{to_state(node.state)}"
    cached = _POLICY_CACHE.get(state_key)
    if cached is not None:
        return cached

    policy = _uniform_policy(legal_moves)
    _POLICY_CACHE[state_key] = policy
    return policy


def _pick_child_for_rollout(children, use_llm_policy):
    if use_llm_policy:
        weights = [max(1e-6, child.prior) for child in children]
        return random.choices(children, weights=weights, k=1)[0]
    return random.choice(children)

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


def train_mcts_during(mcts, training_time, use_llm_policy=True):
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts, use_llm_policy=use_llm_policy)
        current = int(round(time.time() * 1000))
    return mcts


def train_mcts_iterations(mcts=None, iterations=100, use_llm_policy=True):
    for _ in range(max(0, int(iterations))):
        mcts = train_mcts_once(mcts, use_llm_policy=use_llm_policy)
    return mcts


def advance_tree_to_move(root, move, new_state, winner):
    if root is not None and root.children is not None:
        for child in root.children:
            if child.move == move:
                child.parent = None
                return child
    return Node(new_state, winner, move=move, parent=None)


def train_mcts_once(mcts=None, use_llm_policy=True):

    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)

    node = mcts

    # selection
    while node.children is not None:
        if use_llm_policy:
            scores = [child.get_puct(MCTS_C_PUCT) for child in node.children]
            node = node.children[np.argmax(scores)]
            continue

        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    moves = valid_move(node.state)
    if node.winner != 0:
        victorious = node.winner
    elif len(moves) == 0:
        victorious = 0
    else:
        if use_llm_policy:
            policy = _get_policy_for_node(node, moves)
        else:
            policy = _get_pure_policy_for_node(node, moves)

        states = [(play(node.state, move), move) for move in moves]
        node.set_children([
            Node(
                state_winning[0],
                state_winning[1],
                move=move,
                parent=node,
                prior=policy.get(move, 0.0),
            )
            for state_winning, move in states
        ])

        winner_nodes = [child for child in node.children if child.winner]
        if winner_nodes:
            node = winner_nodes[0]
            victorious = node.winner
        else:
            node = _pick_child_for_rollout(node.children, use_llm_policy=use_llm_policy)
            victorious = random_play_improved(node.state)

    # backpropagation
    parent = node
    while parent is not None:
        parent.games += 1
        if victorious != 0 and get_player_to_play(parent.state) != victorious:
            parent.win += 1
        parent = parent.parent

    return mcts
