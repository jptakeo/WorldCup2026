import os
import numpy as np
from cmdstanpy import CmdStanModel
from src.data_prep import preparar_dados_ciclo, carregar_priors_ranking


def treinar_e_salvar(ciclo_nome, modelo_nome, stan_file, df, times, map_times, priors):
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

    # 2. Compilar e Amostrar
    modelo_stan = CmdStanModel(stan_file=stan_file)
    fit = modelo_stan.sample(data=stan_data, chains=4,
                             iter_sampling=500, seed=123)

    # 3. Salvar os draws do NumPy com o nome do modelo específico
    post_draws = fit.stan_variables()
    caminho_saida = f'data/outputs/models/draws_{ciclo_nome}_{modelo_nome}.npz'
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    np.savez_compressed(caminho_saida, **post_draws)
    print(f"Salvo em: {caminho_saida}")


if __name__ == "__main__":
    # Dicionário mapeando um nome amigável para o arquivo .stan correspondente
    modelos_stan = {
        "n_poisson_noranking": "stan_models/n_poisson_noranking.stan",
        "st_dc_ranking": "stan_models/st_dc_ranking.stan",
        "st_dc_noranking": "stan_models/st_dc_noranking.stan",
        "n_dc_ranking": "stan_models/n_dc_ranking.stan",
        "n_dc_noranking": "stan_models/n_dc_noranking.stan",
        "st_poisson_ranking": "stan_models/st_poisson_ranking.stan",
        "st_poisson_noranking": "stan_models/st_poisson_noranking.stan",
        "n_poisson_ranking": "stan_models/n_poisson_ranking.stan"
    }

    print("="*40)
    print("PREPARANDO DADOS DO CICLO 2014")
    print("="*40)
    # A janela exata entre a Copa de 2010 e a Copa de 2014
    df_14, times_14, map_14 = preparar_dados_ciclo(
        'data/raw/results.csv', '2010-06-10', '2014-06-11', aplicar_decaimento=True)

    # Carrega o arquivo de ranking de 2010
    priors_14 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2010.csv', times_14)

    # Roda a bateria de modelos e salva com o prefixo '2014'
    for nome_mod, arquivo_stan in modelos_stan.items():
        treinar_e_salvar('2014', nome_mod, arquivo_stan,
                         df_14, times_14, map_14, priors_14)
#
    # print("="*40)
    # print("PREPARANDO DADOS DO CICLO 2018")
    # print("="*40)
    # df_18, times_18, map_18 = preparar_dados_ciclo(
    #    'data/raw/results.csv', '2014-06-11', '2018-06-13', aplicar_decaimento=True)
    # priors_18 = carregar_priors_ranking(
    #    'data/raw/fifa_ranking_2014.csv', times_18)
    # Loop para rodar os 8 modelos de 2018
    # for nome_mod, arquivo_stan in modelos_stan.items():
    #    treinar_e_salvar('2018', nome_mod, arquivo_stan,
    #                     df_18, times_18, map_18, priors_18)
    # print("="*40)
    # print("PREPARANDO DADOS DO CICLO 2022")
    # print("="*40)
    # df_22, times_22, map_22 = preparar_dados_ciclo(
    #    'data/raw/results.csv', '2018-06-13', '2022-11-20', aplicar_decaimento=True)
    # priors_22 = carregar_priors_ranking(
    #    'data/raw/fifa_ranking_2018.csv', times_22)
    # Loop para rodar os 8 modelos de 2022
    # for nome_mod, arquivo_stan in modelos_stan.items():
    #    treinar_e_salvar('2022', nome_mod, arquivo_stan,
    #                     df_22, times_22, map_22, priors_22)
    print("\n" + "="*40)
    print("PREPARANDO DADOS DO CICLO 2026")
    print("="*40)
    df_26, times_26, map_26 = preparar_dados_ciclo(
        'data/raw/results.csv', '2022-12-19', aplicar_decaimento=True)
    priors_26 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2022.csv', times_26)
    # Loop para rodar os 8 modelos de 2026
    for nome_mod, arquivo_stan in modelos_stan.items():
        treinar_e_salvar('2026', nome_mod, arquivo_stan,
                         df_26, times_26, map_26, priors_26)

    print("\nTodos os modelos foram treinados e salvos com sucesso!")
