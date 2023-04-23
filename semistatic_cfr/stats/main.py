import numpy as np
import pandas as pd
import pathlib

from .inference import get_ci


def calc_confidence_intervals():
    data_path = pathlib.Path('semistatic_cfr', 'experiments', 'tournament_matrix', 'results_action_0.csv')
    data = pd.read_csv(data_path, header=0, index_col=0)

    for player, row in data.iterrows():
        for opponent, datapoint in row.items():
            print(f"Calculating CI for {player} vs. {opponent}")
            mean, history = eval(datapoint)
            data.loc[player][opponent] = get_ci(np.array(history))

    report_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), 'stats_report_action_0.csv')
    data.to_csv(report_path)


if __name__ == "__main__":
    calc_confidence_intervals()
