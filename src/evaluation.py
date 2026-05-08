import numpy as np
import pandas as pd
from scipy.stats import poisson


def calcular_brier_modelo(
    draws, teams_list, caminho_gabarito="data/raw/jogos_2022.csv"
):
    """Evaluate posterior draws with a three-outcome Brier score."""
    t_map = {name: i for i, name in enumerate(teams_list)}

    atk_draws = draws["attack"]
    dfn_draws = draws["defense"]
    eta_draws = draws["eta"]

    usa_dixon_coles = "rho" in draws
    if usa_dixon_coles:
        rho_draws = draws["rho"]

    n_samples = len(eta_draws)
    df_gabarito = pd.read_csv(caminho_gabarito)

    brier_per_sample = np.zeros(n_samples)
    matches_count = 0

    # Masks collapse the score matrix into home win, draw, and away win.
    max_gols = 10
    mask_win = np.tril(np.ones((max_gols + 1, max_gols + 1)), k=-1)
    mask_draw = np.eye(max_gols + 1)
    mask_loss = np.triu(np.ones((max_gols + 1, max_gols + 1)), k=1)

    gols = np.arange(max_gols + 1)

    for row in df_gabarito.itertuples():
        t1, t2 = row.home_team, row.away_team
        g1_real, g2_real = row.home_score, row.away_score

        if t1 not in t_map or t2 not in t_map:
            continue

        matches_count += 1
        idx1, idx2 = t_map[t1], t_map[t2]

        # Expected goals for each posterior draw.
        mu1 = np.exp(atk_draws[:, idx1] - dfn_draws[:, idx2] + eta_draws)
        mu2 = np.exp(atk_draws[:, idx2] - dfn_draws[:, idx1] + eta_draws)

        # Broadcast Poisson PMFs into one score matrix per posterior draw.
        p1 = poisson.pmf(gols[None, :, None], mu1[:, None, None])
        p2 = poisson.pmf(gols[None, None, :], mu2[:, None, None])

        p_matrix = p1 * p2

        # Dixon-Coles adjustment only affects the 0-0, 1-0, 0-1, and 1-1 cells.
        if usa_dixon_coles:
            rho = rho_draws[:, None, None]
            mu1_exp = mu1[:, None, None]
            mu2_exp = mu2[:, None, None]

            p_matrix[:, 0, 0] *= (1 - mu1_exp * mu2_exp * rho)[:, 0, 0]
            p_matrix[:, 1, 0] *= (1 + mu1_exp * rho)[:, 0, 0]
            p_matrix[:, 0, 1] *= (1 + mu2_exp * rho)[:, 0, 0]
            p_matrix[:, 1, 1] *= (1 - rho)[:, 0, 0]

            # Guard against extreme rho values creating invalid probabilities.
            p_matrix = np.clip(p_matrix, a_min=0, a_max=None)

        p_w = np.sum(p_matrix * mask_win, axis=(1, 2))
        p_d = np.sum(p_matrix * mask_draw, axis=(1, 2))
        p_l = np.sum(p_matrix * mask_loss, axis=(1, 2))

        # Truncation at 10 goals leaves tiny tail mass, so normalize the outcomes.
        total = p_w + p_d + p_l
        p_w /= total
        p_d /= total
        p_l /= total

        if g1_real > g2_real:
            y_real = [1, 0, 0]
        elif g1_real == g2_real:
            y_real = [0, 1, 0]
        else:
            y_real = [0, 0, 1]

        erro_quadratico = (
            (p_w - y_real[0]) ** 2 + (p_d - y_real[1]) ** 2 + (p_l - y_real[2]) ** 2
        )
        brier_per_sample += erro_quadratico

    if matches_count > 0:
        brier_per_sample /= matches_count

    return {
        "Brier Mediana": np.median(brier_per_sample),
        "Brier Mean": np.mean(brier_per_sample),
        "Brier Std": np.std(brier_per_sample),
        "IC 2.5%": np.percentile(brier_per_sample, 2.5),
        "IC 97.5%": np.percentile(brier_per_sample, 97.5),
    }
