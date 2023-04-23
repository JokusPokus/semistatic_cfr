"""
Add suboptimal biases to NES strategy, train a BR and save the resulting policy.
"""

import copy
import itertools
import pathlib
import json

import pyspiel

from semistatic_cfr.utils import get_parser, Pickler, Experiment
from semistatic_cfr.policies.base import ActionSelectionBias
from semistatic_cfr.agents import CFRAgent


def calc_and_save():
    parser = get_parser()
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
    parser.add_argument(
        '-e',
        '--experiment',
        type=int,
        default=1000,
        help='Number of playthroughs per experiment',
    )
    args = parser.parse_args()
    game = pyspiel.load_game('leduc_poker', {'players': 2})

    nes_path = pathlib.Path(
        pathlib.Path(__file__).parent.parent.resolve(),
        'NES',
        'state_dict.obj'
    )
    nes_policy = Pickler(nes_path).read()

    save_dir = pathlib.Path(pathlib.Path(__file__).parent.resolve())
    actions, biases = args.actions, args.biases
    for action, bias_weight in itertools.product(actions, biases):
        folder_name = f'action_{action}__' \
                      f'bias_{str(bias_weight).replace(".", "_")}'
        pathlib.Path(save_dir, folder_name).mkdir(parents=True, exist_ok=True)

        biased_policy = copy.deepcopy(nes_policy)
        biased_policy.bias = ActionSelectionBias(action, bias_weight)

        biased_agent = CFRAgent(
            game,
            policy=biased_policy,
            description=f"NES-B ({action}x{bias_weight})"
        )
        br_agent = CFRAgent(
            game,
            static_opponent_policy=biased_policy,
            description=f"BR to NES-B ({action}x{bias_weight})"
        )

        agents = [br_agent, biased_agent]

        experiment = Experiment(
            title=f"best_response_vs_bias_{action}_{bias_weight}",
            game=game,
            players=agents,
            n_rounds=args.experiment,
            report_path=pathlib.Path(pathlib.Path(__file__).parent.resolve(), 'report.txt')
        )

        br_agent.train(n_iterations=args.iterations, experiment=experiment)

        current_policy = br_agent.solver.current_policy()
        average_policy = br_agent.solver.average_policy()

        cur_path = pathlib.Path(save_dir, folder_name, f'current_policy.obj')
        avg_path = pathlib.Path(save_dir, folder_name, f'average_policy.obj')

        Pickler(cur_path).write(current_policy)
        Pickler(avg_path).write(average_policy)

        expl_save_path = pathlib.Path(
            pathlib.Path(__file__).parent.resolve(),
            folder_name,
            'exploitabilities.json'
        )

        with open(expl_save_path, 'w') as f:
            json.dump(br_agent.exploitabilities, f, indent=4)

        payoff_save_path = pathlib.Path(
            pathlib.Path(__file__).parent.resolve(),
            folder_name,
            f'payoffs.json'
        )

        with open(payoff_save_path, 'w') as f:
            br_agent.payoff_history['n_iterations'] = args.experiment
            json.dump(br_agent.payoff_history, f, indent=4)


if __name__ == '__main__':
    calc_and_save()
