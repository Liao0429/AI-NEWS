# Experiments

This directory contains all experiment scripts for reproducing the results in the paper.

## Experiment List

| Experiment | Purpose | File |
|------------|---------|------|
| **1. Sanity Check** | Verify backtest system works correctly | `01_sanity_check.py` |
| **2. Final Experiment** | AI vs Rule comparison (main result) | `final_experiment.py` |

## Running Experiments

### Option 1: Run all experiments (not implemented yet)
```bash
python run_all_experiments.py
```

### Option 2: Run individual experiments
```bash
# Sanity Check
python experiments/01_sanity_check.py

# Final Experiment
python experiments/final_experiment.py
```

## Experiment Outputs

All results are saved to the `results/` directory:
- `results/tables/`: CSV result tables
- `results/figures/`: Visualization figures
- `results/experiment_metadata.json`: Experiment logs
- `results/error_log.csv`: Error logs
