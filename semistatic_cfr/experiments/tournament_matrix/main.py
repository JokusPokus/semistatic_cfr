"""
Experiment: Leduc Poker tournament matrix.
"""

import copy
import itertools
import pathlib

import pandas as pd
import pyspiel

from semistatic_cfr.utils import Experiment, get_parser, Pickler
from semistatic_cfr.agents import Agent
from semistatic_cfr.policies.base import ActionSelectionBias


def run_experiment():
    parser = get_parser()
    parser.add_argument(
        '-r',
        '--rounds',
        type=int,
        help='Number of rounds per match',
        default=1_000_000
    )
    parser.add_argument(
        '-a',
        '--actions',
        type=int,
        nargs='+',
        help='List of action indices to bias',
        required=True
    )
    parser.add_argument(
        '-b',
        '--biases',
        type=float,
        nargs='+',
        help='List of biases to apply to the policy',
        required=True
    )

    args = parser.parse_args()
    game = pyspiel.load_game('leduc_poker', {'players': 2})

    nes_path = pathlib.Path('semistatic_cfr', 'policies', 'NES', 'state_dict.obj')
    nes_policy = Pickler(nes_path).read()

    nes_agent = Agent(game, policy=nes_policy, description="NES agent")
    actions, biases = args.actions, args.biases
    opponents = {
        action: [nes_agent]
        for action in actions
    }

    for action, bias_weight in itertools.product(actions, biases):
        biased_policy = copy.deepcopy(nes_policy)
        biased_policy.bias = ActionSelectionBias(action, bias_weight)
        biased_agent = Agent(
            game,
            policy=biased_policy,
            description=f"Biased NES agent (action: {action}, bias: {bias_weight})"
        )
        opponents[action].append(biased_agent)

    players = {
        action: [Agent(game, policy=nes_policy, description="NES agent")]
        for action in actions
    }

    player_locs = {
        action: [
            f"action_{action}__bias_{str(bias_weight).replace('.', '_')}"
            for bias_weight in biases
        ]
        for action in actions
    }

    for action in actions:
        for loc in player_locs[action]:
            br_path = pathlib.Path('semistatic_cfr', 'policies', 'NES_bias', loc, 'current_policy.obj')
            br_policy = Pickler(br_path).read()

            br_agent = Agent(game, policy=br_policy, description=f"BR agent to {loc}")
            players[action].append(br_agent)

        results = pd.DataFrame(
            columns=[op.description for op in opponents[action]],
            index=[p.description for p in players[action]]
        )
        for player, opponent in itertools.product(players[action], opponents[action]):
            experiment = Experiment(
                title=f"{player.description} vs. {opponent.description}",
                game=game,
                players=[player, opponent],
                n_rounds=args.rounds,
                report_path=pathlib.Path(pathlib.Path(__file__).parent.resolve(), 'report.txt'),
                save_outcomes=True,
            )

            experiment.run()
            results.loc[player.description][opponent.description] = (experiment.p0_average_return, experiment.p0_returns)

        results_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), f'results_action_{action}.csv')
        results.to_csv(results_path)


if __name__ == '__main__':
    run_experiment()
