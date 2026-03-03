import argparse
import random

import numpy as np

from connect4.connect4 import create_grid, get_player_to_play, play, valid_move
from connect4.mcts import Node
from connect4.mcts_ia import (
    advance_tree_to_move,
    get_llm_policy_stats,
    train_mcts_iterations,
)


def utils_print(grid):
    print_grid = grid.astype(str)
    print_grid[print_grid == "-1"] = "X"
    print_grid[print_grid == "1"] = "O"
    print_grid[print_grid == "0"] = " "
    res = str(print_grid).replace("'", "")
    res = res.replace("[[", "[")
    res = res.replace("]]", "]")
    print(" " + res)
    print("  " + " ".join("0123456"))


def _pick_move_from_tree(root, iterations, use_llm_policy):
    root = train_mcts_iterations(
        root, iterations=iterations, use_llm_policy=use_llm_policy
    )
    child, move = root.select_move()
    if move is not None:
        return root, move

    legal_moves = valid_move(root.state)
    if not legal_moves:
        return root, None
    return root, random.choice(legal_moves)


def run_battle_game(iterations_per_move, llm_player):
    grid = create_grid()
    llm_tree = Node(grid, 0, None, None)
    pure_tree = Node(grid, 0, None, None)

    while True:
        player = get_player_to_play(grid)

        if player == llm_player:
            llm_tree, move = _pick_move_from_tree(
                llm_tree, iterations_per_move, use_llm_policy=True
            )
        else:
            pure_tree, move = _pick_move_from_tree(
                pure_tree, iterations_per_move, use_llm_policy=False
            )

        if move is None:
            return 0

        grid, winner = play(grid, move)
        llm_tree = advance_tree_to_move(llm_tree, move, grid, winner)
        pure_tree = advance_tree_to_move(pure_tree, move, grid, winner)

        if winner != 0:
            return winner
        if len(valid_move(grid)) == 0:
            return 0


def run_battle(games, iterations_per_move, llm_side):
    llm_wins = 0
    pure_wins = 0
    draws = 0

    for game_idx in range(games):
        if llm_side == "alternate":
            llm_player = -1 if (game_idx % 2) == 0 else 1
        else:
            llm_player = -1 if llm_side == "first" else 1

        winner = run_battle_game(iterations_per_move, llm_player)
        if winner == 0:
            draws += 1
        elif winner == llm_player:
            llm_wins += 1
        else:
            pure_wins += 1

        print(
            f"game={game_idx + 1}/{games} llm_side={'first' if llm_player == -1 else 'second'} winner={winner}"
        )

    print("\nSummary")
    print(f"games={games}")
    print(f"iterations_per_move={iterations_per_move}")
    print(f"llm_wins={llm_wins}")
    print(f"pure_wins={pure_wins}")
    print(f"draws={draws}")
    print(f"llm_policy_stats={get_llm_policy_stats()}")


def run_interactive(iterations_per_move):
    node = Node(create_grid(), 0, None, None)
    grid = create_grid()
    round_idx = 0
    utils_print(grid)

    while True:
        if (round_idx % 2) == 0:
            move = int(input())
            if move not in valid_move(grid):
                print("Invalid move.")
                continue
        else:
            node, move = _pick_move_from_tree(
                node, iterations_per_move, use_llm_policy=True
            )
            if move is None:
                print("No legal move for AI.")
                break
            print(move)

        grid, winner = play(grid, move)
        node = advance_tree_to_move(node, move, grid, winner)
        utils_print(grid)

        if winner != 0:
            print("Winner:", "X" if winner == -1 else "O")
            break
        if len(valid_move(grid)) == 0:
            print("Draw.")
            break

        round_idx += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Connect4 MCTS runner")
    parser.add_argument(
        "--mode",
        choices=["interactive", "battle"],
        default="interactive",
        help="interactive: play vs AI, battle: pure MCTS vs LLM+MCTS",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="MCTS iterations per move",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games (battle mode only)",
    )
    parser.add_argument(
        "--llm-side",
        choices=["first", "second", "alternate"],
        default="alternate",
        help="Which side the LLM+MCTS agent plays in battle mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "battle":
        run_battle(
            games=max(1, args.games),
            iterations_per_move=max(1, args.iterations),
            llm_side=args.llm_side,
        )
    else:
        run_interactive(iterations_per_move=max(1, args.iterations))
