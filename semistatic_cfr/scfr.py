"""Python implementation of the semistatic counterfactual regret minimization algorithm."""
import numpy as np

from cfr import _CFRSolver, _apply_regret_matching_plus_reset, _update_current_policy


class StaticCFRPlusSolver(_CFRSolver):
    """CFR+ implementation for calculating a best response
    to a static strategy.
    """

    def __init__(self, game, static_policy):
        super(StaticCFRPlusSolver, self).__init__(
            game,
            regret_matching_plus=True,
            alternating_updates=True,
            linear_averaging=True
        )
        self._static_policy = static_policy

    def evaluate_and_update_policy(self):
        """Performs a single step of policy evaluation and policy improvement."""
        self._iteration += 1
        for player in range(self._game.num_players()):
            if player == 0:
                policies = [self._get_infostate_policy, self._static_policy]
            else:
                policies = [self._static_policy, self._get_infostate_policy]

            self._compute_counterfactual_regret_for_player(
                self._root_node,
                policies=policies,
                reach_probabilities=np.ones(self._game.num_players() + 1),
                player=player)

            _apply_regret_matching_plus_reset(self._info_state_nodes)
            _update_current_policy(self._current_policy, self._info_state_nodes)
