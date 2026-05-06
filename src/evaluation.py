import numpy as np
import pandas as pd
from scipy.stats import poisson


def calcular_brier_modelo(draws, teams_list, caminho_gabarito='data/raw/jogos_2022.csv'):
    t_map = {name: i for i, name in enumerate(teams_list)}

    atk_draws = draws['attack']
    dfn_draws = draws['defense']
    eta_draws = draws['eta']

    # Verifica se o modelo atual possui o parâmetro de Dixon-Coles
    usa_dixon_coles = 'rho' in draws
    if usa_dixon_coles:
        rho_draws = draws['rho']

    n_samples = len(eta_draws)
    df_gabarito = pd.read_csv(caminho_gabarito)

    brier_per_sample = np.zeros(n_samples)
    matches_count = 0

    # Matrizes de máscara para somar Vitória (triângulo inferior), Empate (diagonal) e Derrota (triângulo superior)
    max_gols = 10
    mask_win = np.tril(np.ones((max_gols + 1, max_gols + 1)), k=-1)
    mask_draw = np.eye(max_gols + 1)
    mask_loss = np.triu(np.ones((max_gols + 1, max_gols + 1)), k=1)

    # Array de gols [0, 1, 2, ..., 10]
    gols = np.arange(max_gols + 1)

    for row in df_gabarito.itertuples():
        t1, t2 = row.home_team, row.away_team
        g1_real, g2_real = row.home_score, row.away_score

        if t1 not in t_map or t2 not in t_map:
            continue

        matches_count += 1
        idx1, idx2 = t_map[t1], t_map[t2]

        # Vetor de Forças esperadas para as 1000 amostras (shape: 1000)
        mu1 = np.exp(atk_draws[:, idx1] - dfn_draws[:, idx2] + eta_draws)
        mu2 = np.exp(atk_draws[:, idx2] - dfn_draws[:, idx1] + eta_draws)

        # Calcula as PMFs (Probability Mass Function) da Poisson para 0 a 10 gols
        # Usamos [:, None, None] para expandir as dimensões e o Numpy criar as matrizes automaticamente
        # shape: (1000, 11, 1)
        p1 = poisson.pmf(gols[None, :, None], mu1[:, None, None])
        # shape: (1000, 1, 11)
        p2 = poisson.pmf(gols[None, None, :], mu2[:, None, None])

        # Matriz conjunta de probabilidades (shape: 1000, 11, 11)
        p_matrix = p1 * p2

        # --- APLICAÇÃO DO DIXON-COLES (SE EXISTIR NO MODELO) ---
        if usa_dixon_coles:
            rho = rho_draws[:, None, None]  # Expande para shape (1000, 1, 1)
            mu1_exp = mu1[:, None, None]
            mu2_exp = mu2[:, None, None]

            # i=0, j=0
            p_matrix[:, 0, 0] *= (1 - mu1_exp * mu2_exp * rho)[:, 0, 0]
            # i=1, j=0
            p_matrix[:, 1, 0] *= (1 + mu1_exp * rho)[:, 0, 0]
            # i=0, j=1
            p_matrix[:, 0, 1] *= (1 + mu2_exp * rho)[:, 0, 0]
            # i=1, j=1
            p_matrix[:, 1, 1] *= (1 - rho)[:, 0, 0]

            # Previne probabilidades negativas causadas por rhos extremos
            p_matrix = np.clip(p_matrix, a_min=0, a_max=None)

        # Soma as probabilidades de Vitória, Empate e Derrota para cada uma das 1000 amostras
        p_w = np.sum(p_matrix * mask_win, axis=(1, 2))
        p_d = np.sum(p_matrix * mask_draw, axis=(1, 2))
        p_l = np.sum(p_matrix * mask_loss, axis=(1, 2))

        # Normalização (como truncamos em 10 gols, a soma não dá 1.0 exato, então normalizamos)
        total = p_w + p_d + p_l
        p_w /= total
        p_d /= total
        p_l /= total

        # GABARITO
        if g1_real > g2_real:
            y_real = [1, 0, 0]
        elif g1_real == g2_real:
            y_real = [0, 1, 0]
        else:
            y_real = [0, 0, 1]

        # Calcula o Brier Score para as amostras
        erro_quadratico = (p_w - y_real[0])**2 + \
            (p_d - y_real[1])**2 + (p_l - y_real[2])**2
        brier_per_sample += erro_quadratico

    if matches_count > 0:
        brier_per_sample /= matches_count

    return {
        'Brier Mediana': np.median(brier_per_sample),
        'Brier Mean': np.mean(brier_per_sample),
        'Brier Std': np.std(brier_per_sample),
        'IC 2.5%': np.percentile(brier_per_sample, 2.5),   # Limite inferior
        'IC 97.5%': np.percentile(brier_per_sample, 97.5)  # Limite superior
    }
