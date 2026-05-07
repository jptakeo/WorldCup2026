import pandas as pd
import numpy as np
import os
from cmdstanpy import CmdStanModel
from src.data_prep import preparar_dados_ciclo, carregar_priors_ranking
from src.constants import TEAM_MAP_EN_TO_PT, TEAM_MAP_PT_TO_EN
from src.simulate import simular_fase_e_restante
from simulations.sim_2026 import carregar_draws
from src.export_probs import update_html_from_summary


def treinar_e_salvar(ciclo_nome, modelo_nome, stan_file, exe_file, df, times, map_times, priors):
    """
    Função auxiliar que executa o Stan e salva o arquivo compactado.
    Note que agora passamos os dados (df, times) já prontos para não recalcular a cada iteração.
    """
    print(f"\n[{ciclo_nome}] Treinando: {modelo_nome} ...")

    # 1. Preparar o dicionário de dados do Stan
    stan_data = {
        'N': len(df), 'T': len(times),
        'team_i': df['home_team'].map(map_times).values,
        'team_j': df['away_team'].map(map_times).values,
        'y_i': df['home_score'].values.astype(int),
        'y_j': df['away_score'].values.astype(int),
        'game_weight': df['game_weight'].values,
        'prior_strength': priors
    }

    modelo_stan = CmdStanModel(
        stan_file=stan_file,
        exe_file=exe_file
    )

    # 2. Compilar e Amostrar
    modelo_stan = CmdStanModel(stan_file=stan_file)
    fit = modelo_stan.sample(data=stan_data, chains=4,
                             iter_sampling=2500, seed=123)

    # 3. Salvar os draws do NumPy com o nome do modelo específico
    post_draws = fit.stan_variables()
    caminho_saida = f'data/outputs/models/new_draws_{ciclo_nome}_{modelo_nome}.npz'
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    np.savez_compressed(caminho_saida, **post_draws)
    print(f"Salvo em: {caminho_saida}")

if __name__ == "__main__":
    print("\n=== PREPARANDO DADOS ===\n")

    old_results = pd.read_csv('data/raw/results.csv')
    actual_results = pd.read_csv('data/world_cup_results.csv')
    
    if ((actual_results.loc[actual_results['stage'] == 'semi_final', 'home_real'].isna().sum() == 0) & 
        (actual_results.loc[actual_results['stage'] == 'final', 'home_team'].isna().sum() == 0)): 
        stage = 'final'
        matches_to_pred = actual_results.loc[actual_results['stage'].isin(['final', 'third_place'])]
        matches_to_add = actual_results.loc[actual_results['stage'].isin(['round_of_32', 'round_of_16', 'quarter_final', 'semi_final'])]
        matches_to_add["date"] = pd.to_datetime(
            matches_to_add["date"] + "/2026",
            format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        df_to_concat = pd.DataFrame({
            'date': matches_to_add['date'],
            'home_team': matches_to_add['home_team'].map(TEAM_MAP_PT_TO_EN),
            'away_team': matches_to_add['away_team'].map(TEAM_MAP_PT_TO_EN),
            'home_score': matches_to_add['home_real'],
            'away_score': matches_to_add['away_real'],
            'tournament': 'FIFA World Cup',
            'city': None,
            'country': None,
            'neutral': None
        })
        df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
    elif ((actual_results.loc[actual_results['stage'] == 'quarter_final', 'home_real'].isna().sum() == 0) &
            (actual_results.loc[actual_results['stage'] == 'semi_final', 'home_team'].isna().sum() == 0)):
        stage = 'semi_final'
        colunas_para_substituir = ["round_of_32", "round_of_16", "quarter_final", "semi_final"]
        matches_to_pred = actual_results.loc[actual_results['stage'].isin(['semi_final'])]
        matches_to_add = actual_results.loc[actual_results['stage'].isin(['round_of_32', 'round_of_16', 'quarter_final'])]
        matches_to_add["date"] = pd.to_datetime(
            matches_to_add["date"] + "/2026",
            format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        df_to_concat = pd.DataFrame({
            'date': matches_to_add['date'],
            'home_team': matches_to_add['home_team'],
            'away_team': matches_to_add['away_team'],
            'home_score': matches_to_add['home_real'],
            'away_score': matches_to_add['away_real'],
            'tournament': 'FIFA World Cup',
            'city': None,
            'country': None,
            'neutral': None
        })
        df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
    elif ((actual_results.loc[actual_results['stage'] == 'round_of_16', 'home_real'].isna().sum() == 0) &
            (actual_results.loc[actual_results['stage'] == 'quarter_final', 'home_team'].isna().sum() == 0)):
        stage = 'quarter_final'
        colunas_para_substituir = ["round_of_32", "round_of_16", "quarter_final"]
        matches_to_pred = actual_results.loc[actual_results['stage'].isin(['quarter_final'])]
        matches_to_add = actual_results.loc[actual_results['stage'].isin(['round_of_32', 'round_of_16'])]
        matches_to_add["date"] = pd.to_datetime(
            matches_to_add["date"] + "/2026",
            format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        df_to_concat = pd.DataFrame({
            'date': matches_to_add['date'],
            'home_team': matches_to_add['home_team'],
            'away_team': matches_to_add['away_team'],
            'home_score': matches_to_add['home_real'],
            'away_score': matches_to_add['away_real'],
            'tournament': 'FIFA World Cup',
            'city': None,
            'country': None,
            'neutral': None
        })
        df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
    elif ((actual_results.loc[actual_results['stage'] == 'round_of_32', 'home_real'].isna().sum() == 0) &
            (actual_results.loc[actual_results['stage'] == 'round_of_16', 'home_team'].isna().sum() == 0)):
        stage = 'round_of_16'
        colunas_para_substituir = ["round_of_32", "round_of_16"]
        matches_to_pred = actual_results.loc[actual_results['stage'].isin(['round_of_16'])]
        matches_to_add = actual_results.loc[actual_results['stage'] == 'round_of_32']
        matches_to_add["date"] = pd.to_datetime(
            matches_to_add["date"] + "/2026",
            format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        df_to_concat = pd.DataFrame({
            'date': matches_to_add['date'],
            'home_team': matches_to_add['home_team'],
            'away_team': matches_to_add['away_team'],
            'home_score': matches_to_add['home_real'],
            'away_score': matches_to_add['away_real'],
            'tournament': 'FIFA World Cup',
            'city': None,
            'country': None,
            'neutral': None
        })
        df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
    elif ((actual_results.loc[actual_results['stage'] == 'group', 'home_real'].isna().sum() == 0) &
            (actual_results.loc[actual_results['stage'] == 'round_of_32', 'home_team'].isna().sum() == 0)):
        stage = 'round_of_32'
        colunas_para_substituir = ["round_of_32"]
        matches_to_pred = actual_results.loc[actual_results['stage'] == 'round_of_32']
        matches_to_add = actual_results.loc[actual_results['stage'] == 'group']
        matches_to_add["date"] = pd.to_datetime(
            matches_to_add["date"] + "/2026",
            format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        df_to_concat = pd.DataFrame({
            'date': matches_to_add['date'],
            'home_team': matches_to_add['home_team'],
            'away_team': matches_to_add['away_team'],
            'home_score': matches_to_add['home_real'],
            'away_score': matches_to_add['away_real'],
            'tournament': 'FIFA World Cup',
            'city': None,
            'country': None,
            'neutral': None
        })
        df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
    
    df_26, times_26, map_26 = preparar_dados_ciclo(
        'data/new_results.csv',
        '2022-12-19',
        aplicar_decaimento=True
    )

    priors_26 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2022.csv',
        times_26
    )

    print("\n=== COMPILANDO MODELO ===\n")
    stan_file = "stan_models/n_poisson_ranking.stan"
    modelo = CmdStanModel(stan_file=stan_file)

    print("\n=== RETREINANDO MODELO ===\n")
    treinar_e_salvar(
        "2026", 
        "n_poisson_ranking", 
        stan_file, 
        modelo.exe_file,
        df_26,
        times_26,
        map_26,
        priors_26
    )

    nome_modelo = 'new_draws_2026_n_poisson_ranking.npz'
    caminho_modelo = f'data/outputs/models/{nome_modelo}'
    draws_26 = carregar_draws(caminho_modelo)
    matches_to_pred["home_team"] = matches_to_pred["home_team"].map(TEAM_MAP_PT_TO_EN)
    matches_to_pred["away_team"] = matches_to_pred["away_team"].map(TEAM_MAP_PT_TO_EN)
    matches_to_pred.reset_index(drop=True, inplace=True)
    probs, df_summary, df_matches = simular_fase_e_restante(draws_26, times_26, matches_to_pred)

    df_previous_summary = pd.read_csv('data/summary.csv')
    # 1. Define exactly which columns you want to bring over from the previous summary

    # 2. Temporarily set 'team' as the index for both dataframes so they align perfectly
    df_summary = df_summary.set_index('team')
    df_previous_summary = df_previous_summary.set_index('team')

    # 3. Use .update() to overwrite the current columns with the previous ones
    df_summary.update(df_previous_summary[colunas_para_substituir])

    # 4. Reset the index to turn 'team' back into a normal column
    df_summary = df_summary.reset_index()

    if 'position' in df_summary.columns:
        df_summary = df_summary.drop(columns=['position'])
        df_summary.insert(0, 'position', df_summary.index + 1)


    df_summary.to_csv('data/summary.csv', index=False)
    df_matches.to_csv(f'data/probs_{stage}.csv', index=False)
    df_matches.to_csv(f'docs/csv/probs_{stage}.csv', index=False)
    update_html_from_summary()
