"""
Approximate a Nash equilibrium and save the tabular policy as JSON.
"""

import pathlib
import json

import pyspiel

from ...agents import CFRAgent
from semistatic_cfr.utils import get_parser, Pickler


def calc_and_save():
    """Calculate and save the approximate Nash equilibrium
    for Leduc Poker.
    """
    parser = get_parser()
    args = parser.parse_args()
    game = pyspiel.load_game('leduc_poker', {'players': 2})

    cfr_agent = CFRAgent(0, game)
    cfr_agent.train(n_iterations=args.iterations)

    average_policy = cfr_agent.solver.average_policy()

    path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), args.path)

    Pickler(path).write(average_policy)

    stats_save_path = pathlib.Path(
        pathlib.Path(__file__).parent.resolve(),
        'exploitabilities.json'
    )

    with open(stats_save_path, 'w') as f:
        json.dump(cfr_agent.exploitabilities, f, indent=4)


if __name__ == '__main__':
    calc_and_save()
