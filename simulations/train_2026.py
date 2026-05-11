from concurrent.futures import ProcessPoolExecutor, as_completed

from cmdstanpy import CmdStanModel

from model_sel.validate import treinar_e_salvar
from src.data_prep import carregar_priors_ranking, preparar_dados_ciclo

if __name__ == "__main__":
    # Enable additional model variants here when the 2026 training run should
    # compare more than the selected production model.
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
            "exe_file": modelo.exe_file,
        }

    print("\nTodos os modelos compilados!")

    ciclos = []

    print("=" * 50)
    print("PREPARANDO DADOS DO CICLO 2026")
    print("=" * 50)

    df_26, times_26, map_26 = preparar_dados_ciclo(
        "data/raw/results.csv", "2022-11-19", aplicar_decaimento=True
    )

    priors_26 = carregar_priors_ranking("data/raw/fifa_ranking_2022.csv", times_26)

    ciclos.append(("2026", df_26, times_26, map_26, priors_26))

    # One parallel job per configured 2026 model.
    jobs = []

    for ciclo_nome, df, times, map_times, priors in ciclos:
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
                    priors,
                )
            )

    print("\n" + "=" * 50)
    print("INICIANDO TREINAMENTO PARALELO")
    print("=" * 50)

    # Keep worker count conservative because Stan chains are CPU-heavy.
    MAX_WORKERS = 3

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(treinar_e_salvar, *job) for job in jobs]

        for future in as_completed(futures):
            future.result()

    print("\nTodos os modelos foram treinados e salvos!")
