import os
import subprocess

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from simulations.sim_2026 import carregar_draws
from src.constants import TEAM_MAP_PT_TO_EN
from src.data_prep import carregar_priors_ranking, preparar_dados_ciclo
from src.export_probs import update_html_from_summary
from src.simulate import simular_fase_e_restante


def treinar_e_salvar(
    ciclo_nome, modelo_nome, stan_file, exe_file, df, times, map_times, priors
):
    """
    Run an updated 2026 Stan model and save posterior draws for live updates.
    """
    print(f"\n[{ciclo_nome}] Treinando: {modelo_nome} ...")

    # Keep this schema aligned with the data block in the Stan model.
    stan_data = {
        "N": len(df),
        "T": len(times),
        "team_i": df["home_team"].map(map_times).values,
        "team_j": df["away_team"].map(map_times).values,
        "y_i": df["home_score"].values.astype(int),
        "y_j": df["away_score"].values.astype(int),
        "game_weight": df["game_weight"].values,
        "prior_strength": priors,
    }

    modelo_stan = CmdStanModel(stan_file=stan_file, exe_file=exe_file)

    # Recreate the model from source so CmdStan can rebuild stale binaries.
    modelo_stan = CmdStanModel(stan_file=stan_file)
    fit = modelo_stan.sample(data=stan_data, chains=4, iter_sampling=5000, seed=123)

    post_draws = fit.stan_variables()
    caminho_saida = f"data/outputs/models/new_draws_{ciclo_nome}_{modelo_nome}.npz"
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    np.savez_compressed(caminho_saida, **post_draws)
    print(f"Salvo em: {caminho_saida}")


if __name__ == "__main__":
    print("\n=== PREPARANDO DADOS ===\n")

    old_results = pd.read_csv("data/raw/results.csv")
    actual_results = pd.read_csv("data/world_cup_results.csv")

    if actual_results["home_real"].isna().sum() == actual_results.shape[0]:
        print("\n=== TREINANDO MODELO PARA 2026 ===\n")
        subprocess.run(["python", "-m", "simulations.train_2026"], check=True)

        print("\n=== SIMULANDO COPA 2026 ===\n")
        subprocess.run(["python", "-m", "simulations.sim_2026"], check=True)
    else:
        if (
            actual_results.loc[actual_results["stage"] == "semi_final", "home_real"]
            .isna()
            .sum()
            == 0
        ) & (
            actual_results.loc[actual_results["stage"] == "final", "home_team"]
            .isna()
            .sum()
            == 0
        ):
            stage = "final"
            colunas_para_substituir = ["champion"]
            matches_to_pred = actual_results.loc[
                actual_results["stage"].isin(["final"])
            ]
            matches_to_add = actual_results.loc[
                actual_results["stage"].isin(
                    ["round_of_32", "round_of_16", "quarter_final", "semi_final"]
                )
            ]
            matches_to_add["date"] = pd.to_datetime(
                matches_to_add["date"] + "/2026", format="%d/%m/%Y"
            ).dt.strftime("%Y-%m-%d")
            df_to_concat = pd.DataFrame(
                {
                    "date": matches_to_add["date"],
                    "home_team": matches_to_add["home_team"].map(TEAM_MAP_PT_TO_EN),
                    "away_team": matches_to_add["away_team"].map(TEAM_MAP_PT_TO_EN),
                    "home_score": matches_to_add["home_real"],
                    "away_score": matches_to_add["away_real"],
                    "tournament": "FIFA World Cup",
                    "city": None,
                    "country": None,
                    "neutral": None,
                }
            )
            df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
        elif (
            actual_results.loc[actual_results["stage"] == "quarter_final", "home_real"]
            .isna()
            .sum()
            == 0
        ) & (
            actual_results.loc[actual_results["stage"] == "semi_final", "home_team"]
            .isna()
            .sum()
            == 0
        ):
            stage = "semi_final"
            colunas_para_substituir = ["final", "champion"]
            matches_to_pred = actual_results.loc[
                actual_results["stage"].isin(["semi_final"])
            ]
            matches_to_add = actual_results.loc[
                actual_results["stage"].isin(
                    ["round_of_32", "round_of_16", "quarter_final"]
                )
            ]
            matches_to_add["date"] = pd.to_datetime(
                matches_to_add["date"] + "/2026", format="%d/%m/%Y"
            ).dt.strftime("%Y-%m-%d")
            df_to_concat = pd.DataFrame(
                {
                    "date": matches_to_add["date"],
                    "home_team": matches_to_add["home_team"],
                    "away_team": matches_to_add["away_team"],
                    "home_score": matches_to_add["home_real"],
                    "away_score": matches_to_add["away_real"],
                    "tournament": "FIFA World Cup",
                    "city": None,
                    "country": None,
                    "neutral": None,
                }
            )
            df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
        elif (
            actual_results.loc[actual_results["stage"] == "round_of_16", "home_real"]
            .isna()
            .sum()
            == 0
        ) & (
            actual_results.loc[actual_results["stage"] == "quarter_final", "home_team"]
            .isna()
            .sum()
            == 0
        ):
            stage = "quarter_final"
            colunas_para_substituir = ["semifinals", "final", "champion"]
            matches_to_pred = actual_results.loc[
                actual_results["stage"].isin(["quarter_final"])
            ]
            matches_to_add = actual_results.loc[
                actual_results["stage"].isin(["round_of_32", "round_of_16"])
            ]
            matches_to_add["date"] = pd.to_datetime(
                matches_to_add["date"] + "/2026", format="%d/%m/%Y"
            ).dt.strftime("%Y-%m-%d")
            df_to_concat = pd.DataFrame(
                {
                    "date": matches_to_add["date"],
                    "home_team": matches_to_add["home_team"],
                    "away_team": matches_to_add["away_team"],
                    "home_score": matches_to_add["home_real"],
                    "away_score": matches_to_add["away_real"],
                    "tournament": "FIFA World Cup",
                    "city": None,
                    "country": None,
                    "neutral": None,
                }
            )
            df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
        elif (
            actual_results.loc[actual_results["stage"] == "round_of_32", "home_real"]
            .isna()
            .sum()
            == 0
        ) & (
            actual_results.loc[actual_results["stage"] == "round_of_16", "home_team"]
            .isna()
            .sum()
            == 0
        ):
            stage = "round_of_16"
            colunas_para_substituir = [
                "quarterfinals",
                "semifinals",
                "final",
                "champion",
            ]
            matches_to_pred = actual_results.loc[
                actual_results["stage"].isin(["round_of_16"])
            ]
            matches_to_add = actual_results.loc[
                actual_results["stage"] == "round_of_32"
            ]
            matches_to_add["date"] = pd.to_datetime(
                matches_to_add["date"] + "/2026", format="%d/%m/%Y"
            ).dt.strftime("%Y-%m-%d")
            df_to_concat = pd.DataFrame(
                {
                    "date": matches_to_add["date"],
                    "home_team": matches_to_add["home_team"],
                    "away_team": matches_to_add["away_team"],
                    "home_score": matches_to_add["home_real"],
                    "away_score": matches_to_add["away_real"],
                    "tournament": "FIFA World Cup",
                    "city": None,
                    "country": None,
                    "neutral": None,
                }
            )
            df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)
        elif (
            actual_results.loc[actual_results["stage"] == "group", "home_real"]
            .isna()
            .sum()
            == 0
        ) & (
            actual_results.loc[actual_results["stage"] == "round_of_32", "home_team"]
            .isna()
            .sum()
            == 0
        ):
            stage = "round_of_32"
            colunas_para_substituir = [
                "round_of_16",
                "quarterfinals",
                "semifinals",
                "final",
                "champion",
            ]
            matches_to_pred = actual_results.loc[
                actual_results["stage"] == "round_of_32"
            ]
            matches_to_add = actual_results.loc[actual_results["stage"] == "group"]
            matches_to_add["date"] = pd.to_datetime(
                matches_to_add["date"] + "/2026", format="%d/%m/%Y"
            ).dt.strftime("%Y-%m-%d")
            df_to_concat = pd.DataFrame(
                {
                    "date": matches_to_add["date"],
                    "home_team": matches_to_add["home_team"],
                    "away_team": matches_to_add["away_team"],
                    "home_score": matches_to_add["home_real"],
                    "away_score": matches_to_add["away_real"],
                    "tournament": "FIFA World Cup",
                    "city": None,
                    "country": None,
                    "neutral": None,
                }
            )
            df_updated = pd.concat([old_results, df_to_concat], ignore_index=True)

        df_updated.to_csv("data/new_results.csv", index=False)
        df_26, times_26, map_26 = preparar_dados_ciclo(
            "data/new_results.csv", "2022-11-19", aplicar_decaimento=True
        )

        priors_26 = carregar_priors_ranking("data/raw/fifa_ranking_2022.csv", times_26)

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
            priors_26,
        )

        nome_modelo = "new_draws_2026_n_poisson_ranking.npz"
        caminho_modelo = f"data/outputs/models/{nome_modelo}"
        draws_26 = carregar_draws(caminho_modelo)
        matches_to_pred["home_team"] = matches_to_pred["home_team"].map(
            TEAM_MAP_PT_TO_EN
        )
        matches_to_pred["away_team"] = matches_to_pred["away_team"].map(
            TEAM_MAP_PT_TO_EN
        )
        matches_to_pred.reset_index(drop=True, inplace=True)
        if stage == "final":
            pred_third_place = actual_results.loc[
                actual_results["stage"].isin(["third_place"])
            ]
            pred_third_place["home_team"] = pred_third_place["home_team"].map(
                TEAM_MAP_PT_TO_EN
            )
            pred_third_place["away_team"] = pred_third_place["away_team"].map(
                TEAM_MAP_PT_TO_EN
            )
            pred_third_place.reset_index(drop=True, inplace=True)
            probs, df_summary, df_matches = simular_fase_e_restante(
                draws_26, times_26, matches_to_pred
            )
            probs_third_place, _, df_matches_third_place = simular_fase_e_restante(
                draws_26, times_26, pred_third_place
            )
            df_previous_summary = pd.read_csv("data/summary.csv")
            # Align by team before replacing only the newly simulated columns.
            df_summary = df_summary.set_index("team")
            df_previous_summary = df_previous_summary.set_index("team")
            df_previous_summary = df_previous_summary[
                df_previous_summary.index.isin(df_summary.index)
            ]

            df_previous_summary.update(df_summary[colunas_para_substituir])

            # Re-rank the public table after updating champion probability.
            df_previous_summary = df_previous_summary.sort_values(
                by="champion", ascending=False
            )

            df_previous_summary = df_previous_summary.reset_index()
            df_previous_summary["position"] = df_previous_summary.index + 1

            df_previous_summary.to_csv("data/summary.csv", index=False)
            df_previous_summary.to_csv("docs/csv/previsoes/summary.csv", index=False)
            df_matches.to_csv(f"data/probs_{stage}.csv", index=False)
            df_matches.to_csv(f"docs/csv/previsoes/probs_{stage}.csv", index=False)
            df_matches_third_place.to_csv("data/probs_third_place.csv", index=False)
            df_matches_third_place.to_csv(
                "docs/csv/previsoes/probs_third_place.csv", index=False
            )
        else:
            probs, df_summary, df_matches = simular_fase_e_restante(
                draws_26, times_26, matches_to_pred
            )
            df_previous_summary = pd.read_csv("data/summary.csv")
            # Align by team before replacing only the newly simulated columns.
            df_summary = df_summary.set_index("team")
            df_previous_summary = df_previous_summary.set_index("team")
            df_previous_summary = df_previous_summary[
                df_previous_summary.index.isin(df_summary.index)
            ]

            df_previous_summary.update(df_summary[colunas_para_substituir])

            # Re-rank the public table after updating champion probability.
            df_previous_summary = df_previous_summary.sort_values(
                by="champion", ascending=False
            )

            df_previous_summary = df_previous_summary.reset_index()
            df_previous_summary["position"] = df_previous_summary.index + 1

            df_previous_summary.to_csv("data/summary.csv", index=False)
            df_previous_summary.to_csv("docs/csv/previsoes/summary.csv", index=False)
            df_matches.to_csv(f"data/probs_{stage}.csv", index=False)
            df_matches.to_csv(f"docs/csv/previsoes/probs_{stage}.csv", index=False)

    update_html_from_summary()
