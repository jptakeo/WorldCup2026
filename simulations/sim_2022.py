import json
import os

import numpy as np

from src.dashboard import generate_dashboard
from src.data_prep import prepare_cycle_data
from src.simulate import simulate_world_cup_2022


def load_draws(path):
    """Carrega os draws gerados pelo Stan e salvos no treino."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    os.makedirs("data/outputs/results", exist_ok=True)
    os.makedirs("data/outputs/dashboards", exist_ok=True)

    groups_2022 = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    }

    print("\n--- SIMULANDO 2022 ---")

    # Rebuild the team order used when the 2022 model was trained.
    _, teams_22, _ = prepare_cycle_data(
        "data/raw/results.csv", "2018-06-13", "2022-11-20", apply_decay=True
    )

    # Select the saved posterior draws to simulate from.
    model_name = "draws_2022_n_poisson_ranking.npz"
    model_path = f"data/outputs/models/{model_name}"
    print(f"Carregando: {model_path}")
    draws_22 = load_draws(model_path)

    probs_2022 = simulate_world_cup_2022(draws_22, teams_22, groups_2022, n_sim=100000)

    # Dashboard generator expects probabilities grouped by stage.
    json_output_22 = {
        stage: [
            {"team": teams_22[i], "probability": probs_2022[stage][i]}
            for i in range(len(teams_22))
        ]
        for stage in probs_2022.keys()
    }
    with open("data/outputs/results/sim_results_2022.json", "w") as f:
        json.dump(json_output_22, f)

    participants_22 = set(team for teams in groups_2022.values() for team in teams)
    stage_labels_22 = {
        "avancou_grupos": "Fase de Grupos",
        "quarter_finalists": "Oitavas",
        "semi_finalists": "Quartas",
        "finalists": "Semis",
        "champion": "Campeão",
    }
    generate_dashboard(
        "data/outputs/results/sim_results_2022.json",
        "data/outputs/dashboards/dashboard_2022.html",
        stage_labels_22,
        participants_22,
        8,
        "Copa 2022",
        model_name,
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2022.html")
