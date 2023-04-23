# Semistatic Counterfactual Regret Minimization

This repository contains the source code for all simulations and experiments 
conducted in the thesis "Exploiting Suboptimal Strategies Using Counterfactual Regret Minimization"
by Jakob Schmitt.

To reproduce the experiment, you may follow these steps:

### 1. Install dependencies

Install `poetry` and run the following command from the repository's root directory:
```bash
poetry install
```

### 2. Calculate NES

To approximate a NES and save the result as a pickled Python object, run:
```bash
poetry run python3 -m semistatic_cfr.policies.NES.main --iterations 2000
```

### 3. Calculate Best Responses to biased strategies

To approximate Best Responses to biased strategies using Semistatic CFR and save the result 
as a pickled Python object, run:
```bash
poetry run python3 -m semistatic_cfr.policies.NES_bias.main --actions 0 2 --biases 1.5 10 100 --iterations 100 --experiment 20000
```

### 4. Run tournament of strategies

To execute the tournament matrix, run:
```bash
poetry run python3 -m semistatic_cfr.experiment.tournament_matrix.main --actions 0 2 --biases 1.5 10 100 --rounds 1000000
```

### 5. Calculate statistics

To calculate inference statistics on the tournament data, run:
```bash
poetry run python3 -m semistatic_cfr.experiment.tournament_matrix.main --actions 0 2 --biases 1.5 10 100 --rounds 1000000
```
