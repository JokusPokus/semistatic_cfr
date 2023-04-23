"""
Agents to play Liar's Dice.
"""
from typing import Optional

import torch

from open_spiel.python.algorithms import exploitability
from .cfr import CFRPlusSolver, StaticCFRPlusSolver


class Agent:
    """Represents a class of agents, e.g., an AI or a human player."""
    def __init__(
            self,
            game,
            policy: Optional = None,
            description: Optional[str] = None
    ):
        self.game = game
        self.policy = policy
        self.description = description

    def make_bid_for(self, game_state) -> int:
        action_probs = self.policy.action_probabilities(game_state)
        action_probs_tensor = torch.Tensor(list(action_probs.values()))
        action_ind = torch.multinomial(action_probs_tensor, 1).item()
        return list(action_probs.keys())[action_ind]

    def exploitability(self):
        """Calculate the exploitability of the agent's strategy."""
        return exploitability.exploitability(self.game, self.policy)


class CFRAgent(Agent):

    def __init__(
            self,
            game,
            policy=None,
            description=None,
            static_opponent_policy=None,
    ):
        super().__init__(game, policy, description)
        self.exploitabilities = None
        self.static_opponent_policy = static_opponent_policy

        if static_opponent_policy:
            self.solver = StaticCFRPlusSolver(game, static_opponent_policy)
            self.policy = self.solver.average_policy()
            self.exploitabilities = {'average': {}, 'current': {}}
            self.payoff_history = {'average': {}, 'current': {}}

        elif not policy:
            self.solver = CFRPlusSolver(game)
            self.policy = self.solver.average_policy()
            self.exploitabilities = {'average': {}}

    def train(self, n_iterations: int, experiment=None):
        def experiment_scheduled(iteration) -> bool:
            if experiment is None:
                return False

            return (
                iteration < 10
                or (iteration + 1) % 10 == 0
            )

        print(f"Starting training (total iterations: {n_iterations})\n")

        self.exploitabilities['average'][0] = self.exploitability()
        experiment.run(silent=True)
        self.payoff_history['average'][0] = experiment.p0_average_return
        experiment.reset()

        av_pol = self.policy
        self.policy = self.solver.current_policy()
        cur_expl = self.exploitability()
        self.exploitabilities['current'][0] = cur_expl
        experiment.run(silent=True)
        self.payoff_history['current'][0] = experiment.p0_average_return
        experiment.reset()
        self.policy = av_pol

        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")

            self.solver.evaluate_and_update_policy()
            self.policy = self.solver.average_policy()

            av_expl = self.exploitability()
            self.exploitabilities['average'][i+1] = av_expl

            if self.static_opponent_policy:
                av_pol = self.policy
                self.policy = self.solver.current_policy()
                cur_expl = self.exploitability()
                self.exploitabilities['current'][i+1] = cur_expl
                self.policy = av_pol

                if experiment_scheduled(i):
                    experiment.run(silent=True)
                    self.payoff_history['average'][i+1] = experiment.p0_average_return
                    experiment.reset()

                    av_pol = self.policy
                    self.policy = self.solver.current_policy()

                    experiment.run(silent=True)
                    self.payoff_history['current'][i+1] = experiment.p0_average_return
                    experiment.reset()

                    self.policy = av_pol
