import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import preparar_dados_ciclo
from src.evaluation import calcular_brier_modelo


def carregar_draws(caminho):
    """Load compressed posterior draws saved by the Stan training scripts."""
    loaded = np.load(caminho)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    print("\n" + "=" * 45)
    print("AVALIAÇÃO DE PERFORMANCE: BRIER SCORE 2018")
    print("=" * 45)

    # Rebuild the training team order so draw columns line up with teams.
    _, times_18, _ = preparar_dados_ciclo(
        "data/raw/results.csv", "2014-06-11", "2018-06-13", aplicar_decaimento=True
    )

    # Evaluate every saved model for this validation cycle.
    pasta_modelos = "data/outputs/models/"
    if not os.path.exists(pasta_modelos):
        print(
            "Pasta de modelos não encontrada. Certifique-se de rodar train.py primeiro."
        )
        exit()

    arquivos_npz = [
        f
        for f in os.listdir(pasta_modelos)
        if f.startswith("draws_2018_") and f.endswith(".npz")
    ]

    if not arquivos_npz:
        print("Nenhum modelo de 2018 treinado encontrado em data/outputs/models/.")
        exit()

    resultados = []

    for arquivo in arquivos_npz:
        nome_modelo = arquivo.replace("draws_2018_", "").replace(".npz", "")
        print(f"Avaliando: {nome_modelo} ...")

        caminho_completo = os.path.join(pasta_modelos, arquivo)
        draws = carregar_draws(caminho_completo)

        metricas = calcular_brier_modelo(draws, times_18, "data/raw/jogos_2018.csv")

        resultados.append(
            {
                "Modelo": nome_modelo.replace("_", " ").title(),
                "Brier Mediana": metricas["Brier Mediana"],
                "IC Inferior": metricas["IC 2.5%"],
                "IC Superior": metricas["IC 97.5%"],
                "IC 95%": f"[{metricas['IC 2.5%']:.4f}  -  {metricas['IC 97.5%']:.4f}]",
            }
        )

    # Lower Brier scores indicate better calibrated three-outcome predictions.
    df_resultados = (
        pd.DataFrame(resultados).sort_values("Brier Mediana").reset_index(drop=True)
    )

    print("\n--- RANKING FINAL DE PRECISÃO ---")
    print(df_resultados.to_string())

    print("\nGerando gráfico de sobreposição...")
    plt.figure(figsize=(10, 6))
    plt.style.use("ggplot")

    # Reverse the order so the best model appears at the top of the plot.
    df_plot = df_resultados.sort_values("Brier Mediana", ascending=False).reset_index(
        drop=True
    )

    for i in range(len(df_plot)):
        mediana = df_plot.loc[i, "Brier Mediana"]
        inf = df_plot.loc[i, "IC Inferior"]
        sup = df_plot.loc[i, "IC Superior"]

        plt.plot([inf, sup], [i, i], color="#3498db", linewidth=5, alpha=0.8)
        plt.plot(mediana, i, "ko", markersize=8)

    # Reference line for the best median score.
    melhor_mediana = df_plot["Brier Mediana"].min()
    plt.axvline(
        melhor_mediana,
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

    caminho_grafico = "data/outputs/results/comparacao_brier_2018.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"Gráfico salvo com sucesso em: {caminho_grafico}")

    caminho_csv = "data/outputs/results/brier_score_2018.csv"
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"\nResultados salvos em: {caminho_csv}")
