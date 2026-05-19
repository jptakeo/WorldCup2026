# Source Code Guide

`src/` contains the reusable code behind the World Cup forecasting workflow.
The code is split between data access/preparation, Stan-posterior simulation,
frequency-model simulation, evaluation, dashboard generation, and exports for
the static website.

## Top-Level Modules

| Module | Purpose |
| --- | --- |
| `constants.py` | Shared paths, 2026 groups, bracket rules, score labels, team-name mappings, and display labels. |
| `fetch_kaggle_dataset.py` | Downloads the Kaggle international-results dataset and refreshes `data/results.csv`, `data/shootouts.csv`, and `data/goalscorers.csv`. |
| `data_prep.py` | Builds cycle-specific training frames for Stan: filters low-volume teams, computes match importance/time-decay weights, maps teams to Stan IDs, and loads ranking priors. |
| `simulate.py` | Simulates tournaments from Stan posterior draws and exports match/stage probability tables. |
| `evaluation.py` | Computes Brier-score summaries for saved posterior draws against known match results. |
| `dashboard.py` | Generates standalone D3 HTML dashboards from simulation JSON files. |
| `export_probs.py` | Exports current-phase score probability matrices, builds public stage summaries, and updates `docs/chances.html`. |
| `forces.py` | Plots posterior team-strength distributions from saved Stan draws. |

## Public Helper Names

The reusable Stan-path helpers use standardized English names:

| Module | Main Helpers |
| --- | --- |
| `data_prep.py` | `filter_low_volume_teams()`, `get_importance_factor()`, `prepare_cycle_data()`, `load_ranking_priors()` |
| `simulate.py` | `simulate_matches()`, `simulate_world_cup_2022()`, `simulate_world_cup_2026()`, `simulate_stage_and_remaining()` |
| `evaluation.py` | `calculate_model_brier()` |
| `dashboard.py` | `generate_dashboard()` |

## `freq_model/`

`freq_model/` is the faster maximum-likelihood modeling path. It fits a
Dixon-Coles attack/defense model and simulates the full 48-team 2026 tournament.

| Module | Purpose |
| --- | --- |
| `data_classes.py` | Parameter containers used by tournament simulation. |
| `dixon_coles_base.py` | Shared match-probability and match-simulation API. |
| `model.py` | Data weighting and L-BFGS-B fitting for the Dixon-Coles model. |
| `tournament.py` | Group-stage, best-third allocation, knockout bracket, and Monte Carlo simulation logic. |
| `utils.py` | Score-matrix math, phase detection, team-name resolution, and result loading. |
| `simulate.py` | Command-line wrapper for fitting, simulating, exporting probabilities, and updating site HTML. |

## Data Flow

1. `fetch_kaggle_dataset.py` refreshes the match-result CSVs.
2. `data_prep.py` or `freq_model.model` prepares training data.
3. Stan scripts or `freq_model.model` fit team strengths.
4. `simulate.py` or `freq_model.tournament` runs Monte Carlo tournaments.
5. `export_probs.py` writes public CSVs and updates the static website.

## Maintenance Notes

- Keep generated CSV/JSON/HTML outputs out of source modules.
- Preserve the English team names used by models and use the mappings in
  `constants.py` for Portuguese display names.
- Use English for Python identifiers. Portuguese labels, output filenames, and
  website-facing paths should remain unchanged unless the static site is updated
  at the same time.
- When adding a new tournament phase, update the constants, simulator, export
  code, and website CSV expectations together.
