"""
Extension of Policy class to add action selection biases.
"""

from collections import namedtuple
from typing import Optional

from open_spiel.python.policy import TabularPolicy


ActionSelectionBias = namedtuple('ActionSelectionBias', ['action', 'bias'])


class CustomTabularPolicy(TabularPolicy):
    def __init__(
            self,
            policy: TabularPolicy,
            bias: Optional[ActionSelectionBias] = None,
    ):
        self.__dict__ = policy.__dict__
        self.bias = bias

    def action_probabilities(self, state, player_id=None) -> None:
        probs = super().action_probabilities(state, player_id)

        if self.bias is None:
            return probs

        action, bias_weight = self.bias
        biased_action_is_illegal = action not in probs
        n_legal_actions = len(probs)

        if biased_action_is_illegal or n_legal_actions == 1:
            return probs

        probs[action] *= bias_weight

        prob_sum = sum(probs.values())

        for action in probs:
            probs[action] /= prob_sum

        return probs
