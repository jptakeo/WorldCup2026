import os
import json
import numpy as np
import pandas as pd
from src.data_prep import preparar_dados_ciclo
from src.simulation import simular_copa_2026
from src.dashboard import generate_dashboard


def carregar_draws(caminho):
    """Carrega os draws gerados pelo Stan e salvos no treino."""
    loaded = np.load(caminho)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    os.makedirs('data/outputs/results', exist_ok=True)
    os.makedirs('data/outputs/dashboards', exist_ok=True)

    grupos_2026 = {
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
        "L": ["England", "Croatia", "Ghana", "Panama"]
    }

    print("\n--- SIMULANDO 2026 ---")

    # 1. Pega os nomes dos times em ordem alfabética
    _, times_26, _ = preparar_dados_ciclo(
        'data/raw/results.csv', '2022-12-19', aplicar_decaimento=True)

    # 2. Carrega o modelo que você quiser testar (mude o nome do arquivo se necessário)
    nome_modelo = 'draws_2026_n_poisson_ranking.npz'
    caminho_modelo = f'data/outputs/models/{nome_modelo}'
    print(f"Carregando: {caminho_modelo}")
    draws_26 = carregar_draws(caminho_modelo)

    # 3. Roda a simulação
    probs_2026 = simular_copa_2026(
        draws_26, times_26, grupos_2026, n_sim=100000)

    # 4. Salva o JSON
    json_output_26 = {fase: [{'team': times_26[i], 'probability': probs_2026[fase][i]}
                             for i in range(len(times_26))] for fase in probs_2026.keys()}
    with open('data/outputs/results/sim_results_2026.json', 'w') as f:
        json.dump(json_output_26, f)

    # 5. Gera o Dashboard HTML
    participantes_26 = set(team for teams in grupos_2026.values()
                           for team in teams)
    fases_26 = {
        "avancou_grupos": "Fase de Grupos", "round_of_32": "16 Avos",
        "round_of_16": "Oitavas", "quarter_finalists": "Quartas",
        "semi_finalists": "Semis", "finalists": "Final", "champion": "Campeão"
    }
    generate_dashboard('data/outputs/results/sim_results_2026.json',
                       'data/outputs/dashboards/dashboard_2026.html', fases_26, participantes_26, 12, "Copa 2026", nome_modelo)

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2026.html")
