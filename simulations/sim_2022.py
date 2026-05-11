import json
import os

import numpy as np

from src.dashboard import generate_dashboard
from src.data_prep import preparar_dados_ciclo
from src.simulate import simular_copa_2022


def carregar_draws(caminho):
    """Carrega os draws gerados pelo Stan e salvos no treino."""
    loaded = np.load(caminho)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    os.makedirs("data/outputs/results", exist_ok=True)
    os.makedirs("data/outputs/dashboards", exist_ok=True)

    grupos_2022 = {
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
    _, times_22, _ = preparar_dados_ciclo(
        "data/raw/results.csv", "2018-06-13", "2022-11-20", aplicar_decaimento=True
    )

    # Select the saved posterior draws to simulate from.
    nome_modelo = "draws_2022_n_poisson_ranking.npz"
    caminho_modelo = f"data/outputs/models/{nome_modelo}"
    print(f"Carregando: {caminho_modelo}")
    draws_22 = carregar_draws(caminho_modelo)

    probs_2022 = simular_copa_2022(draws_22, times_22, grupos_2022, n_sim=100000)

    # Dashboard generator expects probabilities grouped by stage.
    json_output_22 = {
        fase: [
            {"team": times_22[i], "probability": probs_2022[fase][i]}
            for i in range(len(times_22))
        ]
        for fase in probs_2022.keys()
    }
    with open("data/outputs/results/sim_results_2022.json", "w") as f:
        json.dump(json_output_22, f)

    participantes_22 = set(team for teams in grupos_2022.values() for team in teams)
    fases_22 = {
        "avancou_grupos": "Fase de Grupos",
        "quarter_finalists": "Oitavas",
        "semi_finalists": "Quartas",
        "finalists": "Semis",
        "champion": "Campeão",
    }
    generate_dashboard(
        "data/outputs/results/sim_results_2022.json",
        "data/outputs/dashboards/dashboard_2022.html",
        fases_22,
        participantes_22,
        8,
        "Copa 2022",
        nome_modelo,
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2022.html")
