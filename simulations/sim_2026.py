import json
import os

import numpy as np

from src.dashboard import generate_dashboard
from src.data_prep import prepare_cycle_data
from src.export_probs import update_html_from_summary
from src.simulate import simulate_world_cup_2026


def load_draws(path):
    """Carrega os draws gerados pelo Stan e salvos no treino."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    os.makedirs("data/outputs/results", exist_ok=True)
    os.makedirs("data/outputs/dashboards", exist_ok=True)

    groups_2026 = {
        "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
        "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
        "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "I": ["France", "Senegal", "Iraq", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    }

    # Rebuild the team order used when the 2026 model was trained.
    _, teams_26, _ = prepare_cycle_data(
        "data/raw/results.csv", "2022-11-19", apply_decay=True
    )

    # Select the saved posterior draws to simulate from.
    model_name = "draws_2026_n_poisson_ranking.npz"
    model_path = f"data/outputs/models/{model_name}"
    print(f"Carregando: {model_path}")
    draws_26 = load_draws(model_path)

    probs_2026 = simulate_world_cup_2026(draws_26, teams_26, groups_2026, n_sim=100_000)

    # Dashboard generator expects probabilities grouped by stage.
    json_output_26 = {
        stage: [
            {"team": teams_26[i], "probability": probs_2026[stage][i]}
            for i in range(len(teams_26))
        ]
        for stage in probs_2026.keys()
    }
    with open("data/outputs/results/sim_results_2026.json", "w") as f:
        json.dump(json_output_26, f)

    participants_26 = set(team for teams in groups_2026.values() for team in teams)
    stage_labels_26 = {
        "avancou_grupos": "Fase de Grupos",
        "round_of_32": "16 Avos",
        "round_of_16": "Oitavas",
        "quarter_finalists": "Quartas",
        "semi_finalists": "Semis",
        "finalists": "Final",
        "champion": "Campeão",
    }
    generate_dashboard(
        "data/outputs/results/sim_results_2026.json",
        "data/outputs/dashboards/dashboard_2026.html",
        stage_labels_26,
        participants_26,
        12,
        "Copa 2026",
        model_name,
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2026.html")

    update_html_from_summary()
