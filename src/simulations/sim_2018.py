from __future__ import annotations

import json
import os

from src.data import prepare_cycle_data
from src.model.bayesian import load_draws
from src.output import generate_dashboard
from src.simulations.utils import simulate_world_cup_2022

if __name__ == "__main__":
    os.makedirs("data/outputs/results", exist_ok=True)
    os.makedirs("data/outputs/dashboards", exist_ok=True)

    # Official 2018 World Cup groups.
    groups_2018 = {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    }

    print("\n--- SIMULANDO 2018 ---")

    # Rebuild the team order used when the 2018 model was trained.
    _, teams_18, _ = prepare_cycle_data(
        "data/raw/results.csv", "2014-06-11", "2018-06-13", apply_decay=True
    )

    # Select the saved posterior draws to simulate from.
    model_name = "draws_2018_n_poisson_ranking.npz"
    model_path = f"data/outputs/models/{model_name}"
    print(f"Carregando: {model_path}")
    draws_18 = load_draws(model_path)

    # The 2018 and 2022 tournaments share the same 32-team bracket format.
    probs_2018 = simulate_world_cup_2022(draws_18, teams_18, groups_2018, n_sim=100_000)

    # Dashboard generator expects probabilities grouped by stage.
    json_output_18 = {
        stage: [
            {"team": teams_18[i], "probability": probs_2018[stage][i]}
            for i in range(len(teams_18))
        ]
        for stage in probs_2018
    }

    with open("data/outputs/results/sim_results_2018.json", "w") as f:
        json.dump(json_output_18, f)

    participants_18 = {team for teams in groups_2018.values() for team in teams}

    # Dashboard labels use public-facing stage names.
    stage_labels_18 = {
        "avancou_grupos": "Oitavas de Final",
        "quarter_finalists": "Quartas de Final",
        "semi_finalists": "Semifinais",
        "finalists": "Finalistas",
        "champion": "Campeão",
    }

    generate_dashboard(
        "data/outputs/results/sim_results_2018.json",
        "data/outputs/dashboards/dashboard_2018.html",
        stage_labels_18,
        participants_18,
        8,
        "Copa do Mundo 2018",
        model_name,
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2018.html")
