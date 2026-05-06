import os
import numpy as np
from cmdstanpy import CmdStanModel
from src.data_prep import preparar_dados_ciclo, carregar_priors_ranking
from concurrent.futures import ProcessPoolExecutor, as_completed
from model_sel.validate import treinar_e_salvar

if __name__ == "__main__":
    # Dicionário mapeando um nome amigável para o arquivo .stan correspondente
    modelos_stan = {
        # "n_poisson_noranking": "stan_models/n_poisson_noranking.stan",
        # "st_dc_ranking": "stan_models/st_dc_ranking.stan",
        # "st_dc_noranking": "stan_models/st_dc_noranking.stan",
        # "n_dc_ranking": "stan_models/n_dc_ranking.stan",
        # "n_dc_noranking": "stan_models/n_dc_noranking.stan",
        # "st_poisson_ranking": "stan_models/st_poisson_ranking.stan",
        # "st_poisson_noranking": "stan_models/st_poisson_noranking.stan",
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
    print("PREPARANDO DADOS DO CICLO 2026")
    print("=" * 50)

    df_26, times_26, map_26 = preparar_dados_ciclo(
        'data/raw/results.csv',
        '2022-12-19',
        aplicar_decaimento=True
    )

    priors_26 = carregar_priors_ranking(
        'data/raw/fifa_ranking_2022.csv',
        times_26
    )

    ciclos.append(
        ('2026', df_26, times_26, map_26, priors_26)
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