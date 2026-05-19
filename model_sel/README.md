# Model Selection and Validation

`model_sel/` contains scripts for training and comparing the Stan model family
on completed World Cup cycles.

## Files

| File | Purpose |
| --- | --- |
| `validate.py` | Trains all configured Stan model variants for the 2018 and 2022 cycles and stores posterior draws in `data/outputs/models/`. |
| `evaluate_2018.py` | Loads saved 2018 draws, evaluates them against `data/raw/jogos_2018.csv`, writes a Brier-score CSV, and saves a comparison plot. |
| `evaluate_2022.py` | Loads saved 2022 draws, evaluates them against the 2022 answer key, writes a Brier-score CSV, and saves a comparison plot. |

## Shared Naming

The model-selection scripts use standardized English identifiers:

- `train_and_save()` in `validate.py` runs one Stan model/cycle pair and writes
  posterior draws;
- `load_draws()` in the evaluation scripts loads compressed `.npz` posterior
  files;
- `calculate_model_brier()` from `src.evaluation` computes Brier-score
  summaries;
- `prepare_cycle_data()` and `load_ranking_priors()` from `src.data_prep`
  provide training data and ranking priors.

## Expected Inputs

- `data/raw/results.csv`
- `data/raw/fifa_ranking_2014.csv`
- `data/raw/fifa_ranking_2018.csv`
- `data/raw/jogos_2018.csv`
- `data/raw/jogos_2022.csv`
- Stan model files in `stan_models/`

## Outputs

- `data/outputs/models/draws_<cycle>_<model>.npz`
- `data/outputs/results/brier_score_2018.csv`
- `data/outputs/results/brier_score_2022.csv`
- `data/outputs/results/comparacao_brier_2018.png`
- `data/outputs/results/comparacao_brier_2022.png`

## Notes

These scripts run CmdStan sampling and can be slow. They are script entry points,
not import-only library modules. Keep model names aligned with the files in
`stan_models/` and with the naming convention documented there.

Python identifiers should stay in English. The historical answer-key filenames
`jogos_2018.csv` and `jogos_2022.csv`, and generated plot names such as
`comparacao_brier_*.png`, are retained as data/output contracts.
