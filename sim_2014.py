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

    # Grupos oficiais da Copa do Mundo de 2014
    grupos_2014 = {
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        # Nota: Verifique se no seu csv está Ivory Coast ou Côte d'Ivoire
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"]
    }

    print("\n--- SIMULANDO 2014 ---")

    # 1. Pega os nomes dos times em ordem alfabética (usando as datas do ciclo de 2014)
    _, times_14, _ = preparar_dados_ciclo(
        'data/raw/results.csv', '2010-07-12', '2014-06-11', aplicar_decaimento=True)

    # 2. Carrega o modelo que você quiser testar (ajuste o nome se necessário)
    nome_modelo = 'draws_2014_n_poisson_ranking.npz'
    caminho_modelo = f'data/outputs/models/{nome_modelo}'
    print(f"Carregando: {caminho_modelo}")
    draws_14 = carregar_draws(caminho_modelo)

    # 3. Roda a simulação de 100 mil universos paralelos (reaproveitando a função de 2014)
    probs_2014 = simular_copa_2022(
        draws_14, times_14, grupos_2014, n_sim=100000)

    # 4. Salva o JSON
    json_output_14 = {fase: [{'team': times_14[i], 'probability': probs_2014[fase][i]}
                             for i in range(len(times_14))] for fase in probs_2014.keys()}

    with open('data/outputs/results/sim_results_2014.json', 'w') as f:
        json.dump(json_output_14, f)

    # 5. Gera o Dashboard HTML
    participantes_14 = set(team for teams in grupos_2014.values()
                           for team in teams)

    # Ajustei os nomes para ficarem mais precisos visualmente no painel
    fases_14 = {
        "avancou_grupos": "Oitavas de Final",
        "quarter_finalists": "Quartas de Final",
        "semi_finalists": "Semifinais",
        "finalists": "Finalistas",
        "champion": "Campeão"
    }

    generate_dashboard(
        'data/outputs/results/sim_results_2014.json',
        'data/outputs/dashboards/dashboard_2014.html',
        fases_14,
        participantes_14,
        8,
        "Copa do Mundo 2014",
        nome_modelo
    )

    print("Sucesso! Dashboard gerado em data/outputs/dashboards/dashboard_2014.html")
