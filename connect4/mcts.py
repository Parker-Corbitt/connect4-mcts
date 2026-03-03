import numpy as np

class Node:

    def __init__(self, state, winning, move, parent, prior=1.0):
        self.parent = parent
        self.move = move
        self.win = 0
        self.games = 0
        self.children = None
        self.state = state
        self.winner = winning
        self.prior = float(prior)
        self.depth = 0 if parent is None else parent.depth + 1

    def set_children(self, children):
        self.children = children

    def get_q(self):
        if self.games == 0:
            return 0.0
        return self.win / self.games

    def get_uct(self):
        if self.games == 0:
            return None
        return self.get_q() + np.sqrt(2*np.log(self.parent.games)/self.games)

    def get_puct(self, c_puct=1.25):
        parent_games = 1 if self.parent is None else max(1, self.parent.games)
        exploration = c_puct * self.prior * np.sqrt(parent_games) / (1 + self.games)
        return self.get_q() + exploration


    def select_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None, None

        winners = [child for child in self.children if child.winner]
        if len(winners) > 0:
            return winners[0], winners[0].move

        scores = [child.get_q() for child in self.children]
        best_child = self.children[np.argmax(scores)]
        return best_child, best_child.move


    def get_children_with_move(self, move):
        if self.children is None:
            return None
        for child in self.children:
            if child.move == move:
                return child

        raise Exception('Not existing child')
