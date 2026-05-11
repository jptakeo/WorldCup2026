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

    # Official 2018 World Cup groups.
    grupos_2018 = {
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
    _, times_18, _ = preparar_dados_ciclo(
        "data/raw/results.csv", "2014-06-11", "2018-06-13", aplicar_decaimento=True
    )

    # Select the saved posterior draws to simulate from.
    nome_modelo = "draws_2018_n_poisson_ranking.npz"
    caminho_modelo = f"data/outputs/models/{nome_modelo}"
    print(f"Carregando: {caminho_modelo}")
    draws_18 = carregar_draws(caminho_modelo)

    # The 2018 and 2022 tournaments share the same 32-team bracket format.
    probs_2018 = simular_copa_2022(draws_18, times_18, grupos_2018, n_sim=100000)

    # Dashboard generator expects probabilities grouped by stage.
    json_output_18 = {
        fase: [
            {"team": times_18[i], "probability": probs_2018[fase][i]}
            for i in range(len(times_18))
        ]
        for fase in probs_2018.keys()
    }

    with open("data/outputs/results/sim_results_2018.json", "w") as f:
        json.dump(json_output_18, f)

    participantes_18 = set(team for teams in grupos_2018.values() for team in teams)

    # Dashboard labels use public-facing stage names.
    fases_18 = {
        "avancou_grupos": "Oitavas de Final",
        "quarter_finalists": "Quartas de Final",
        "semi_finalists": "Semifinais",
        "finalists": "Finalistas",
        "champion": "Campeão",
    }

    generate_dashboard(
        "data/outputs/results/sim_results_2018.json",
        "data/outputs/dashboards/dashboard_2018.html",
        fases_18,
        participantes_18,
        8,
        "Copa do Mundo 2018",
        nome_modelo,
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2018.html")
