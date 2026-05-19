# Simulation Scripts

`simulations/` contains executable scripts that connect prepared data, saved
Stan draws, tournament simulators, dashboards, and site exports.

## Files

| File | Purpose |
| --- | --- |
| `train_2026.py` | Trains the selected Stan model for the 2026 cycle and saves posterior draws. |
| `sim_2026.py` | Loads 2026 posterior draws, simulates the 48-team tournament, writes JSON/CSV outputs, generates a dashboard, and updates the site probability table. |
| `update_2026.py` | Rebuilds 2026 probabilities after known World Cup results are added to `data/world_cup_results.csv`. |
| `sim_2022.py` | Historical 2022 simulation from saved Stan draws. |
| `sim_2018.py` | Historical 2018 simulation from saved Stan draws. |

## Shared Naming

The simulation scripts use English helper names imported from `src/` and
`model_sel/`:

- `prepare_cycle_data()` and `load_ranking_priors()` from `src.data_prep`;
- `simulate_world_cup_2022()`, `simulate_world_cup_2026()`, and
  `simulate_stage_and_remaining()` from `src.simulate`;
- local `load_draws()` helpers for `.npz` posterior files;
- `train_and_save()` from `model_sel.validate` for Stan sampling jobs.

## Typical 2026 Flow

```bash
python -m simulations.train_2026
python -m simulations.sim_2026
```

After real 2026 results are entered:

```bash
python -m simulations.update_2026
```

## Generated Outputs

- `data/outputs/models/*.npz`
- `data/outputs/results/sim_results_*.json`
- `data/outputs/dashboards/dashboard_*.html`
- `data/summary.csv`
- `data/probs_*.csv`
- `docs/csv/previsoes/*.csv`
- updated data embedded in `docs/chances.html`

## Maintenance Notes

- Keep tournament group dictionaries explicit in the script that uses them.
- Keep Python variables and helper names in English. Public labels and generated
  website paths may remain Portuguese.
- Avoid changing output filenames unless the static website is updated at the
  same time.
- Use `sim_2018.py` and `sim_2022.py` as historical validation helpers; the main
  active workflow is the 2026 path.
