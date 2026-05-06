import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_prep import preparar_dados_ciclo
from src.evaluation import calcular_brier_modelo


def carregar_draws(caminho):
    loaded = np.load(caminho)
    return {key: loaded[key] for key in loaded.files}


if __name__ == "__main__":
    print("\n" + "="*45)
    print("AVALIAÇÃO DE PERFORMANCE: BRIER SCORE 2014")
    print("="*45)

    # 1. Lista os times para garantir o mesmo mapeamento de índices do treino
    _, times_14, _ = preparar_dados_ciclo(
        'data/raw/results.csv', '2010-07-12', '2014-06-11', aplicar_decaimento=True)

    # 2. Varre a pasta de modelos para encontrar todos os draws de 2014
    pasta_modelos = 'data/outputs/models/'
    if not os.path.exists(pasta_modelos):
        print("Pasta de modelos não encontrada. Certifique-se de rodar train.py primeiro.")
        exit()

    arquivos_npz = [f for f in os.listdir(pasta_modelos) if f.startswith(
        'draws_2014_') and f.endswith('.npz')]

    if not arquivos_npz:
        print("Nenhum modelo de 2014 treinado encontrado em data/outputs/models/.")
        exit()

    resultados = []

    # 3. Calcula o Brier Score para cada modelo
    for arquivo in arquivos_npz:
        nome_modelo = arquivo.replace('draws_2014_', '').replace('.npz', '')
        print(f"Avaliando: {nome_modelo} ...")

        caminho_completo = os.path.join(pasta_modelos, arquivo)
        draws = carregar_draws(caminho_completo)

        metricas = calcular_brier_modelo(
            draws, times_14, 'data/raw/jogos_2014.csv')

        # Consolida para o DataFrame
        resultados.append({
            'Modelo': nome_modelo.replace('_', ' ').title(),
            'Brier Mediana': metricas['Brier Mediana'],
            'IC Inferior': metricas['IC 2.5%'],
            'IC Superior': metricas['IC 97.5%'],
            'IC 95%': f"[{metricas['IC 2.5%']:.4f}  -  {metricas['IC 97.5%']:.4f}]"
        })

    # 4. Cria e exibe a tabela de ranking (menor Brier é melhor)
    df_resultados = pd.DataFrame(resultados).sort_values(
        'Brier Mediana').reset_index(drop=True)

    print("\n--- RANKING FINAL DE PRECISÃO ---")
    print(df_resultados.to_string())

    # 5. Gera o Gráfico Visual de Sobreposição
    print("\nGerando gráfico de sobreposição...")
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    # Inverte para o melhor modelo ficar no topo do gráfico
    df_plot = df_resultados.sort_values(
        'Brier Mediana', ascending=False).reset_index(drop=True)

    for i in range(len(df_plot)):
        mediana = df_plot.loc[i, 'Brier Mediana']
        inf = df_plot.loc[i, 'IC Inferior']
        sup = df_plot.loc[i, 'IC Superior']

        # Desenha a linha do intervalo e o ponto da mediana
        plt.plot([inf, sup], [i, i], color='#3498db', linewidth=5, alpha=0.8)
        plt.plot(mediana, i, 'ko', markersize=8)  # 'ko' = black dot

    # Adiciona uma linha tracejada vertical na mediana do melhor modelo
    melhor_mediana = df_plot['Brier Mediana'].min()
    plt.axvline(melhor_mediana, color='red', linestyle='--', alpha=0.6,
                label='Mediana do Melhor Modelo')

    plt.yticks(range(len(df_plot)), df_plot['Modelo'], fontsize=10)
    plt.xlabel('Brier Score (Menor é Melhor)', fontsize=12)
    plt.title(
        'Comparação de Modelos: Intervalos de Credibilidade (95%)', fontsize=14)
    plt.legend()
    plt.tight_layout()

    caminho_grafico = 'data/outputs/results/comparacao_brier_2014.png'
    plt.savefig(caminho_grafico, dpi=300)
    print(f"Gráfico salvo com sucesso em: {caminho_grafico}")

    # Salva o resultado em CSV para documentação
    caminho_csv = 'data/outputs/results/brier_score_2014.csv'
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"\nResultados salvos em: {caminho_csv}")
