# World Cup 2026

Predictive modeling, simulation, evaluation, and static-site assets for the FIFA
World Cup 2026 project developed around EMAp/FGV work on football forecasting.

The repository contains the full path from international match data to published
probability tables: data download and preparation, Bayesian Stan models,
frequency/Dixon-Coles models, Monte Carlo tournament simulators, validation
scripts for the 2018 and 2022 World Cups, generated CSV/JSON/HTML outputs, and
the public static website under `docs/`.

## What This Repository Does

This project estimates national-team strength from recent international match
results and uses those estimates to simulate World Cup tournaments. The main
outputs are:

- qualification and title probabilities by team;
- match-level scoreline probabilities;
- deterministic/static bracket probability exports;
- model comparison outputs such as Brier score tables and plots;
- HTML dashboards and a Portuguese static site for the public-facing project.

There are two modeling tracks in the repo:

- `src/freq_model/`: a maximum-likelihood Dixon-Coles/frequency-model pipeline
  used for fast fitting and simulation.
- `stan_models/`, `src/data_prep.py`, `src/simulate.py`, `simulations/`, and
  `model_sel/`: Stan-based Bayesian workflows with posterior draws saved as
  compressed NumPy files.

## Repository Layout

| Path | Description |
| --- | --- |
| `README.md` | This guide. |
| `pyproject.toml` | Python package metadata, core dependencies, Ruff, mypy, pytest, and coverage configuration. |
| `requirements.txt` | Additional lightweight requirements used by the Stan-oriented scripts, including `cmdstanpy`. |
| `uv.lock` | Locked dependency resolution for `uv`. |
| `.python-version` | Python version expected by the project tooling. |
| `.pre-commit-config.yaml` | Pre-commit hook configuration. |
| `Makefile` | Common setup, lint, formatting, data download, and cleanup commands for macOS/Linux. |
| `setup.ps1` | Windows PowerShell setup helper. |
| `LICENSE` | MIT license. |
| `.github/` | GitHub repository configuration. |
| `data/` | Current datasets, derived probability tables, simulation summaries, and generated outputs. |
| `data/raw/` | Raw historical match and FIFA ranking inputs used for training and validation. |
| `data/outputs/models/` | Saved Stan posterior draws as `.npz` files. These can be large generated artifacts. |
| `data/outputs/results/` | JSON simulation results and evaluation tables/plots. |
| `data/outputs/dashboards/` | Generated standalone HTML dashboards. |
| `docs/` | Static website served by GitHub Pages or any static file server. |
| `docs/csv/` | CSV files consumed by the website pages and JavaScript visualizations. |
| `docs/css/`, `docs/js/`, `docs/images/` | Website styling, client-side behavior, fonts, flags, team photos, mascots, shirts, and visual assets. |
| `model_sel/` | Model validation, training, and Brier-score evaluation scripts for historical tournaments. |
| `notebooks/` | Exploratory notebooks for data cleaning, model experimentation, and examples. |
| `simulations/` | Script entry points for training and simulating the 2018, 2022, and 2026 tournaments. |
| `src/` | Main Python source code for data access, modeling, simulation, export, evaluation, and dashboards. |
| `src/freq_model/` | Faster frequency-model implementation: data classes, Dixon-Coles likelihood, utilities, and tournament simulator. |
| `stan_models/` | Stan model definitions and compiled model binaries. |

## Main Source Modules

| File | Role |
| --- | --- |
| `src/constants.py` | Central constants: data paths, tournament weights, 2026 groups, bracket rules, score labels, team-name mappings, and display labels. |
| `src/fetch_kaggle_dataset.py` | Downloads the Kaggle international football results dataset, copies `results.csv`, `shootouts.csv`, and `goalscorers.csv` into `data/`, and sorts results by date. |
| `src/data_prep.py` | Prepares cycle-specific training data for Stan models: filters low-volume teams, creates tournament/time weights, maps teams to integer IDs, and loads FIFA ranking priors. |
| `src/simulate.py` | Vectorized simulation utilities for Stan posterior draws, including 2022-style tournaments, 2026 group/bracket simulation, current-phase simulation, and CSV exports. |
| `src/evaluation.py` | Computes Brier scores for fitted posterior draws against known World Cup match results. |
| `src/dashboard.py` | Generates D3-based standalone HTML dashboards from simulation JSON outputs. |
| `src/export_probs.py` | Exports match score probability matrices, stage summaries, and updates the website probability table in `docs/chances.html`. |
| `src/forces.py` | Generates a posterior team-strength distribution plot from saved Stan draws. |

## Frequentist Model Track

The `src/freq_model/` package implements a fast Dixon-Coles-style model and a
2026 tournament simulator.

| File | Role |
| --- | --- |
| `data_classes.py` | Structured containers for fitted model parameters and posterior-like parameter series. |
| `dixon_coles_base.py` | Shared match-probability logic for Dixon-Coles models. |
| `model.py` | Loads and weights match data, fits attack/defense/home/rho parameters with L-BFGS-B, and exposes `build_model()`. |
| `tournament.py` | Full 48-team World Cup 2026 simulator: group standings, best third-place allocation, round-of-32 bracket, knockout rounds, known-result handling, and vectorized Monte Carlo loops. |
| `utils.py` | Name resolution, probability-matrix helpers, World Cup result loading, and phase detection. |
| `simulate.py` | CLI-style entry point for fitting the frequency model, running tournament simulations, exporting probabilities, and updating the site. |

## Stan Model Track

The Stan workflow uses model files in `stan_models/`, prepares weighted match
data with `src/data_prep.py`, runs CmdStan through `cmdstanpy`, saves posterior
draws to `data/outputs/models/`, and then simulates tournaments from those
draws with `src/simulate.py`.

The Stan-oriented Python API uses English names. The main helpers are
`prepare_cycle_data()`, `load_ranking_priors()`, `simulate_matches()`,
`simulate_world_cup_2022()`, `simulate_world_cup_2026()`,
`simulate_stage_and_remaining()`, `calculate_model_brier()`, `load_draws()`,
and `train_and_save()`.

The model names encode two choices:

- `poisson` vs. `dc`: independent Poisson scoring vs. Dixon-Coles low-score
  correction.
- `ranking` vs. `noranking`: whether FIFA ranking information is included as a
  prior-strength feature.
- `n_` vs. `st_`: normal vs. Student-t style parameterization in the Stan model
  family.

Historical validation scripts in `model_sel/` train/evaluate 2018 and 2022
models, while `simulations/train_2026.py`, `simulations/sim_2026.py`, and
`simulations/update_2026.py` handle the 2026 workflow.

## Data

The project uses the
[International Football Results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
dataset from Kaggle. The upstream dataset name mentions 2017, but the dataset is
updated over time. The downloader expects:

| File | Description |
| --- | --- |
| `results.csv` | International match results with teams, scores, tournament, venue, and neutrality fields. |
| `shootouts.csv` | Penalty shootout outcomes. |
| `goalscorers.csv` | Individual goal-scoring records. |

Additional project data includes:

- `data/world_cup_results.csv`: 2026 schedule/results file used to fix known
  World Cup results and detect the current phase.
- `data/summary.csv`: current public probability summary by team.
- `data/probs_*.csv`: match-level probability exports for the current or named
  phase.
- `data/chaveamento_probs.csv`: bracket probability export.
- `data/raw/fifa_ranking_*.csv`: ranking inputs used as Stan priors.
- `docs/csv/previsoes/*.csv`: website-facing copies of prediction and bracket
  data.

To download or refresh the Kaggle dataset:

```bash
make run-data
```

On Windows:

```powershell
uv run python src/fetch_kaggle_dataset.py
```

## Setup

### macOS / Linux

```bash
git clone <repo-url>
cd WorldCup2026
make local
```

### Windows PowerShell

```powershell
git clone <repo-url>
cd WorldCup2026
.\setup.ps1
```

Both setup paths are intended to install `uv`, install the Python version from
`.python-version`, create a virtual environment, install dependencies, and set up
pre-commit hooks.

For Stan workflows, ensure CmdStan is installed and available to `cmdstanpy`.
The repository includes compiled binaries in `stan_models/`, but they are
platform-specific generated artifacts and may need to be rebuilt locally.

## Common Commands

| Task | macOS/Linux | Windows PowerShell |
| --- | --- | --- |
| Setup | `make local` | `.\setup.ps1` |
| Lint and typecheck | `make check` | `uv run ruff check src/; uv run mypy src/` |
| Format | `make format` | `uv run ruff format src/` |
| Run pre-commit hooks | `make test` | `uv run pre-commit run --all-files` |
| Download Kaggle data | `make run-data` | `uv run python src/fetch_kaggle_dataset.py` |
| Clean generated caches | `make clean` | Remove `.venv`, cache folders, and build artifacts manually |

Useful script entry points:

```bash
# Export current-phase score probabilities with the frequency model
uv run python src/export_probs.py --wc-results data/world_cup_results.csv

# Train the selected 2026 Stan model
uv run python -m simulations.train_2026

# Simulate 2026 from saved Stan posterior draws and update site CSV/HTML outputs
uv run python -m simulations.sim_2026

# Re-train/re-simulate after adding known 2026 match results
uv run python -m simulations.update_2026

# Train/evaluate Stan models for 2018 and 2022 validation cycles
uv run python -m model_sel.validate
uv run python -m model_sel.evaluate_2018
uv run python -m model_sel.evaluate_2022
```

The active Stan scripts and shared modules use standardized English variable
and function names. Portuguese filenames and website-facing paths such as
`docs/csv/previsoes/` are kept as part of the public static-site contract.

## Website

The public site lives in `docs/` and is written as static HTML/CSS/JavaScript in
Portuguese. It includes:

- `docs/index.html`: project landing page;
- `docs/modelos.html`: model descriptions;
- `docs/previsoes.html`: match predictions;
- `docs/chances.html`: team advancement/title chances;
- `docs/calendario.html`: schedule page;
- `docs/vis.html`: retrospective visualizations;
- `docs/equipe.html`: team page;
- `docs/css/`, `docs/js/`, `docs/images/`, and `docs/csv/`: styling, behavior,
  assets, and data consumed by the pages.

Because the site is static, it can be viewed by opening `docs/index.html` in a
browser or by serving `docs/` with any static server.

## Generated Outputs

Several files in `data/`, `docs/csv/`, and `data/outputs/` are generated by the
training and simulation scripts. In particular:

- model draw files: `data/outputs/models/*.npz`;
- simulation JSON: `data/outputs/results/sim_results_*.json`;
- dashboards: `data/outputs/dashboards/dashboard_*.html`;
- public probability tables: `data/summary.csv`,
  `docs/csv/previsoes/partidas.csv`, and
  `docs/csv/previsoes/chaveamento_probs.csv`;
- evaluation tables and plots: `data/outputs/results/brier_score_*.csv` and
  `data/outputs/results/comparacao_brier_*.png`.

When changing the modeling code, regenerate the relevant outputs before updating
the website.

## Notes for Contributors

- Keep source changes in `src/`, `simulations/`, `model_sel/`, or
  `stan_models/` separate from generated output changes when possible.
- Use `data/world_cup_results.csv` to add known 2026 match results and let the
  update scripts propagate changes into summaries and site files.
- Run formatting and checks before committing source changes.
- Treat `2022Cup/` as an archive unless intentionally maintaining the legacy
  2022 site or simulations.
