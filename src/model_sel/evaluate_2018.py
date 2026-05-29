import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import calculate_model_brier
from src.data import prepare_cycle_data


def load_draws(path):
    """Load compressed posterior draws saved by the Stan training scripts."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    print("\n" + "=" * 45)
    print("AVALIAÇÃO DE PERFORMANCE: BRIER SCORE 2018")
    print("=" * 45)

    # Rebuild the training team order so draw columns line up with teams.
    _, teams_18, _ = prepare_cycle_data(
        "data/raw/results.csv", "2014-06-11", "2018-06-13", apply_decay=True
    )

    # Evaluate every saved model for this validation cycle.
    models_dir = "data/outputs/models/"
    if not os.path.exists(models_dir):
        print(
            "Pasta de modelos não encontrada. Certifique-se de rodar train.py primeiro."
        )
        exit()

    npz_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("draws_2018_") and f.endswith(".npz")
    ]

    if not npz_files:
        print("Nenhum modelo de 2018 treinado encontrado em data/outputs/models/.")
        exit()

    results = []

    for file_name in npz_files:
        model_name = file_name.replace("draws_2018_", "").replace(".npz", "")
        print(f"Avaliando: {model_name} ...")

        full_path = os.path.join(models_dir, file_name)
        draws = load_draws(full_path)

        metrics = calculate_model_brier(draws, teams_18, "data/raw/jogos_2018.csv")

        results.append(
            {
                "Modelo": model_name.replace("_", " ").title(),
                "Brier Mediana": metrics["Brier Mediana"],
                "IC Inferior": metrics["IC 2.5%"],
                "IC Superior": metrics["IC 97.5%"],
                "IC 95%": f"[{metrics['IC 2.5%']:.4f}  -  {metrics['IC 97.5%']:.4f}]",
            }
        )

    # Lower Brier scores indicate better calibrated three-outcome predictions.
    results_df = (
        pd.DataFrame(results).sort_values("Brier Mediana").reset_index(drop=True)
    )

    print("\n--- RANKING FINAL DE PRECISÃO ---")
    print(results_df.to_string())

    print("\nGerando gráfico de sobreposição...")
    plt.figure(figsize=(10, 6))
    plt.style.use("ggplot")

    # Reverse the order so the best model appears at the top of the plot.
    df_plot = results_df.sort_values("Brier Mediana", ascending=False).reset_index(
        drop=True
    )

    for i in range(len(df_plot)):
        median = df_plot.loc[i, "Brier Mediana"]
        lower = df_plot.loc[i, "IC Inferior"]
        upper = df_plot.loc[i, "IC Superior"]

        plt.plot([lower, upper], [i, i], color="#3498db", linewidth=5, alpha=0.8)
        plt.plot(median, i, "ko", markersize=8)

    # Reference line for the best median score.
    best_median = df_plot["Brier Mediana"].min()
    plt.axvline(
        best_median,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Mediana do Melhor Modelo",
    )

    plt.yticks(range(len(df_plot)), df_plot["Modelo"], fontsize=10)
    plt.xlabel("Brier Score (Menor é Melhor)", fontsize=12)
    plt.title("Comparação de Modelos: Intervalos de Credibilidade (95%)", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plot_path = "data/outputs/results/comparacao_brier_2018.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico salvo com sucesso em: {plot_path}")

    csv_output_path = "data/outputs/results/brier_score_2018.csv"
    results_df.to_csv(csv_output_path, index=False)
    print(f"\nResultados salvos em: {csv_output_path}")
