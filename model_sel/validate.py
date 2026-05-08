import os
import numpy as np
from cmdstanpy import CmdStanModel
from src.data_prep import preparar_dados_ciclo, carregar_priors_ranking
from concurrent.futures import ProcessPoolExecutor, as_completed


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

    print("=" * 50)
    print("PRECOMPILANDO MODELOS STAN")
    print("=" * 50)

    compiled_models = {}

    for nome_mod, stan_file in modelos_stan.items():

        print(f"Compilando {nome_mod} ...")

        modelo = CmdStanModel(stan_file=stan_file)

        compiled_models[nome_mod] = {
            "stan_file": stan_file,
            "exe_file": modelo.exe_file
        }

    print("\nTodos os modelos compilados!")

    ciclos = []

    print("=" * 50)
    print("PREPARANDO DADOS DO CICLO 2018")
    print("=" * 50)

    df_18, times_18, map_18 = preparar_dados_ciclo(
        'data/raw/results.csv',
        '2014-06-11',
        '2018-06-13',
        aplicar_decaimento=True
    )

    priors_18 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2014.csv',
        times_18
    )

    ciclos.append(
        ('2018', df_18, times_18, map_18, priors_18)
    )

    print("=" * 50)
    print("PREPARANDO DADOS DO CICLO 2022")
    print("=" * 50)

    df_22, times_22, map_22 = preparar_dados_ciclo(
        'data/raw/results.csv',
        '2018-06-13',
        '2022-11-20',
        aplicar_decaimento=True
    )

    priors_22 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2018.csv',
        times_22
    )

    ciclos.append(
        ('2022', df_22, times_22, map_22, priors_22)
    )

    # ==========================================
    # Criar lista de jobs
    # ==========================================

    jobs = []

    for (
        ciclo_nome,
        df,
        times,
        map_times,
        priors
    ) in ciclos:

        for nome_mod in modelos_stan.keys():

            jobs.append(
                (
                    ciclo_nome,
                    nome_mod,
                    compiled_models[nome_mod]["stan_file"],
                    compiled_models[nome_mod]["exe_file"],
                    df,
                    times,
                    map_times,
                    priors
                )
            )

    print("\n" + "=" * 50)
    print("INICIANDO TREINAMENTO PARALELO")
    print("=" * 50)

    # Ajuste conforme número de núcleos
    MAX_WORKERS = 3

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                treinar_e_salvar,
                *job
            )
            for job in jobs
        ]

        for future in as_completed(futures):
            future.result()

    print("\nTodos os modelos foram treinados e salvos!")