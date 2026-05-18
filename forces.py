import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_prep import preparar_dados_ciclo

_, times_26, _ = preparar_dados_ciclo(
    'data/raw/results.csv',
    '2022-11-20',
    aplicar_decaimento=True
)
caminho_npz = 'data/outputs/models/draws_2026_n_poisson_ranking.npz'
dados_stan = np.load(caminho_npz)
atk_samples = dados_stan['attack']
def_samples = dados_stan['defense']

strength_samples = atk_samples + def_samples

medias_forca = np.mean(strength_samples, axis=0)
top_indices = np.argsort(medias_forca)[::-1][:20]

dados_plot = []
for idx in top_indices:
    equipe = times_26[idx]
    # Pega nas 10.000 amostras específicas desta equipa
    amostras_da_equipa = strength_samples[:, idx]

    for val in amostras_da_equipa:
        dados_plot.append({'Time': equipe, 'Strength': val})

df_plot = pd.DataFrame(dados_plot)

plt.figure(figsize=(14, 8))

# OPÇÃO 1: VIOLIN PLOT (Mostra o formato gordinho da distribuição de probabilidade)
sns.violinplot(
    x='Strength',
    y='Time',
    data=df_plot,
    palette='viridis',
    inner='quartile',  # Desenha as linhas dos quartis dentro do violino
    linewidth=1.2
)

# Estilização
plt.title('Distribuição da Força (Strength) - Top 20 Seleções',
          fontsize=16, fontweight='bold')
plt.xlabel('Força Total (Atk - Def)', fontsize=12)
plt.ylabel('Seleção', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('data/outputs/results/distribuicao_strength.png', dpi=150)
