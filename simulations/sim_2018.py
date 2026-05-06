import os
import json
import numpy as np
from src.data_prep import preparar_dados_ciclo
from src.simulation import simular_copa_2022
from src.dashboard import generate_dashboard


def carregar_draws(caminho):
    """Carrega os draws gerados pelo Stan e salvos no treino."""
    loaded = np.load(caminho)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    os.makedirs('data/outputs/results', exist_ok=True)
    os.makedirs('data/outputs/dashboards', exist_ok=True)

    # Grupos oficiais da Copa do Mundo de 2018
    grupos_2018 = {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"]
    }

    print("\n--- SIMULANDO 2018 ---")

    # 1. Pega os nomes dos times em ordem alfabética (usando as datas do ciclo de 2018)
    _, times_18, _ = preparar_dados_ciclo(
        'data/raw/results.csv', '2014-07-14', '2018-06-13', aplicar_decaimento=True)

    # 2. Carrega o modelo que você quiser testar (ajuste o nome se necessário)
    nome_modelo = 'draws_2018_n_poisson_ranking.npz'
    caminho_modelo = f'data/outputs/models/{nome_modelo}'
    print(f"Carregando: {caminho_modelo}")
    draws_18 = carregar_draws(caminho_modelo)

    # 3. Roda a simulação de 100 mil universos paralelos (reaproveitando a função de 2022)
    probs_2018 = simular_copa_2022(
        draws_18, times_18, grupos_2018, n_sim=100000)

    # 4. Salva o JSON
    json_output_18 = {fase: [{'team': times_18[i], 'probability': probs_2018[fase][i]}
                             for i in range(len(times_18))] for fase in probs_2018.keys()}

    with open('data/outputs/results/sim_results_2018.json', 'w') as f:
        json.dump(json_output_18, f)

    # 5. Gera o Dashboard HTML
    participantes_18 = set(team for teams in grupos_2018.values()
                           for team in teams)

    # Ajustei os nomes para ficarem mais precisos visualmente no painel
    fases_18 = {
        "avancou_grupos": "Oitavas de Final",
        "quarter_finalists": "Quartas de Final",
        "semi_finalists": "Semifinais",
        "finalists": "Finalistas",
        "champion": "Campeão"
    }

    generate_dashboard(
        'data/outputs/results/sim_results_2018.json',
        'data/outputs/dashboards/dashboard_2018.html',
        fases_18,
        participantes_18,
        8,
        "Copa do Mundo 2018",
        nome_modelo
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2018.html")
