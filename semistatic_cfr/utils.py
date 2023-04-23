"""
Utilities for model training and experiments.
"""

import argparse
import datetime
import pathlib
import pickle
from typing import List, Optional

import numpy as np

from .policies.base import CustomTabularPolicy


class Experiment:
    """Orchestrates one round of a given game."""
    def __init__(
            self,
            title: str,
            game,
            players: List,
            n_rounds: int = 10_000,
            report_path: Optional[pathlib.Path] = None,
            save_outcomes: bool = False
    ):
        self.title = title
        self.game = game

        self._original_players = players
        self._players = players.copy()
        self._n_rounds = n_rounds
        self._report_path = report_path
        self._p0_wins = 0
        self._p1_wins = 0
        self.p0_average_return = 0
        self.p0_returns = []
        self._save_outcomes = save_outcomes

    def run(self, silent=False):
        if not silent:
            print(f"Starting experiment: {self.title}")

        for i in range(self._n_rounds):
            if i == self._n_rounds // 2:
                self._swap_player_order()

            self._play_through(i)

        if not silent:
            print(f"Game over. Player 0 win percentage: {self._p0_percentage}% "
                  f"(vs. {self._p1_percentage}%)")
            print(f"Average return for Player 0: {round(self.p0_average_return, 2)}$")

            if self._report_path:
                self._write_report()

    def reset(self):
        self._players = self._original_players.copy()
        self._players[0].id, self._players[1].id = 0, 1
        self._p0_wins = 0
        self._p1_wins = 0
        self.p0_average_return = 0
        self.p0_returns = []

    @property
    def _p0_percentage(self):
        return round(self._p0_wins * 100 / self._n_rounds, 2)

    @property
    def _p1_percentage(self):
        return round(self._p1_wins * 100 / self._n_rounds, 2)

    def _swap_player_order(self):
        self._players.reverse()

    def _play_through(self, i):
        state = self.game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                self._sample_chance_outcome(state)
            else:
                action = self._players[state.current_player()].make_bid_for(state)
                state.apply_action(action)

        self._record_return_stats(i, state)

    def _record_return_stats(self, i, state):
        returns = state.returns()
        player_0_ind = 0 if i < self._n_rounds // 2 else 1
        if self._save_outcomes:
            self.p0_returns.append(returns[player_0_ind])
        self.p0_average_return += returns[player_0_ind] / self._n_rounds
        self._p0_wins += returns[player_0_ind] > returns[1 - player_0_ind]
        self._p1_wins += returns[player_0_ind] < returns[1 - player_0_ind]


    @staticmethod
    def _sample_chance_outcome(state):
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)

    def _write_report(self):
        with open(self._report_path, 'a+') as f:
            f.write(
                f"""
*****
Meta info:
Time: {datetime.datetime.now()}
Title: {self.title}
Game: {self.game}
Player 0: {self._original_players[0].description}
Player 1: {self._original_players[1].description}
*****

Results after {self._n_rounds} rounds:

Player 0 win percentage: {self._p0_percentage}%
Player 1 win percentage: {self._p1_percentage}%
Average return for Player 0: $ {self.p0_average_return}
""")


class Pickler:
    """Saves and loads policies."""
    def __init__(self, path):
        self.path = path

    def write(self, policy):
        with open(self.path, 'wb') as file:
            pickle.dump(policy, file, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self):
        with open(self.path, 'rb') as file:
            policy = pickle.load(file)

        return CustomTabularPolicy(policy)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="From where to load the model"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations"
    )
    return parser
