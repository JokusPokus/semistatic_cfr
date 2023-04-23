"""Calculate inference statistics for game simulation outcomes."""

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

import numpy as np


def get_ci(data: np.array, interval: float = 0.95) -> tuple:
    return bs.bootstrap(data, stat_func=bs_stats.mean, alpha=1-interval)
