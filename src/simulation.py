import numpy as np

# O Segredo da Velocidade: Calculamos o fatorial na mão para evitar a lentidão das bibliotecas!
# Vai de 0! (1) até 10! (3628800)
FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800])


def simular_jogos(mu1, mu2, rho_draws=None, n_sim=100000, max_gols=10):
    if rho_draws is None:
        # Modo Rápido Poisson Puro
        return np.random.poisson(mu1), np.random.poisson(mu2)

    if mu1.ndim == 1:
        mu1, mu2, rho_draws = mu1[:, None], mu2[:, None], rho_draws[:, None]

    rho = rho_draws
    n_jogos = mu1.shape[1]
    gols = np.arange(max_gols + 1)

    # 1. PMF VETORIZADA MATEMÁTICA (100x mais rápido que Scipy)
    mu1_exp = mu1[:, :, None, None]
    mu2_exp = mu2[:, :, None, None]

    p1 = np.exp(-mu1_exp) * (mu1_exp **
                             gols[None, None, :, None]) / FACTORIALS[None, None, :, None]
    p2 = np.exp(-mu2_exp) * (mu2_exp **
                             gols[None, None, None, :]) / FACTORIALS[None, None, None, :]

    p_matrix = p1 * p2

    # 2. Correção Dixon-Coles
    rho_exp = rho[:, :, None, None]
    p_matrix[:, :, 0, 0] *= (1 - mu1_exp * mu2_exp * rho_exp)[:, :, 0, 0]
    p_matrix[:, :, 1, 0] *= (1 + mu2_exp * rho_exp)[:,
                                                    :, 0, 0]
    p_matrix[:, :, 0, 1] *= (1 + mu1_exp * rho_exp)[:,
                                                    :, 0, 0]
    p_matrix[:, :, 1, 1] *= (1 - rho_exp)[:, :, 0, 0]

    p_matrix = np.clip(p_matrix, a_min=0, a_max=None)

    # 3. Sorteio Super Rápido
    p_flat = p_matrix.reshape(n_sim, n_jogos, -1)
    p_flat /= p_flat.sum(axis=2, keepdims=True)

    cum_p = np.cumsum(p_flat, axis=2)
    rand_vals = np.random.rand(n_sim, n_jogos, 1)

    idx_placar = np.argmax(cum_p > rand_vals, axis=2)

    g1 = idx_placar // (max_gols + 1)
    g2 = idx_placar % (max_gols + 1)

    return g1, g2


def simular_copa_2026(post_draws, teams_list, grupos, n_sim=100000):
    atk_draws, dfn_draws, eta_draws = post_draws['attack'], post_draws['defense'], post_draws['eta']
    usa_dixon_coles = 'rho' in post_draws
    rho_draws = post_draws['rho'] if usa_dixon_coles else None

    n_samples, n_teams = len(eta_draws), len(teams_list)
    t_to_idx = {name: i for i, name in enumerate(teams_list)}
    g_indices = np.array([[t_to_idx[t] for t in ts] for ts in grupos.values()])

    idx_samples = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[idx_samples], dfn_draws[idx_samples]
    et = eta_draws[idx_samples].reshape(-1, 1)
    rho = rho_draws[idx_samples] if usa_dixon_coles else None

    stats = {f: np.zeros(n_teams) for f in ["avancou_grupos", "round_of_32",
                                            "round_of_16", "quarter_finalists", "semi_finalists", "finalists", "champion"]}

    # --- FASE DE GRUPOS ---
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pts, sg, gp = np.zeros((n_sim, 12, 4)), np.zeros(
        (n_sim, 12, 4)), np.zeros((n_sim, 12, 4))

    for p1, p2 in pairs:
        i1, i2 = g_indices[:, p1], g_indices[:, p2]
        l1 = np.exp(atk[:, i1] - dfn[:, i2] + et)
        l2 = np.exp(atk[:, i2] - dfn[:, i1] + et)
        rho_exp = np.repeat(
            rho[:, None], 12, axis=1) if usa_dixon_coles else None

        g1, g2 = simular_jogos(l1, l2, rho_exp, n_sim)

        pts[:, :, p1] += (g1 > g2)*3 + (g1 == g2)
        pts[:, :, p2] += (g2 > g1)*3 + (g1 == g2)
        sg[:, :, p1] += (g1 - g2)
        sg[:, :, p2] += (g2 - g1)
        gp[:, :, p1] += g1
        gp[:, :, p2] += g2

    sort_val = pts * 1000000 + sg * 1000 + gp
    ranks = np.argsort(-sort_val, axis=2)

    classificados = np.zeros((n_sim, 32), dtype=int)
    terceiros_val = np.zeros((n_sim, 12))

    for g in range(12):
        classificados[:, g*2] = g_indices[g, ranks[:, g, 0]]
        classificados[:, g*2+1] = g_indices[g, ranks[:, g, 1]]
        terceiros_val[:, g] = sort_val[np.arange(n_sim), g, ranks[:, g, 2]]

    # FIM DO FOR LENTO: Vetorização dos melhores terceiros colocados!
    best_3rd_idx = np.argsort(-terceiros_val, axis=1)[:, :8]
    row_idx = np.arange(n_sim)[:, None]
    classificados[:, 24:] = g_indices[best_3rd_idx,
                                      ranks[row_idx, best_3rd_idx, 2]]

    # FIM DO FOR LENTO: Função contadora instantânea!
    def contar_fase(array_fase, nome_fase):
        unique, counts = np.unique(array_fase, return_counts=True)
        stats[nome_fase][unique] += counts

    contar_fase(classificados, "avancou_grupos")

    # --- MATA-MATA ---
    def rodada(competidores):
        n = competidores.shape[1] // 2
        a, b = competidores[:, 0::2], competidores[:, 1::2]
        la = np.exp(atk[np.arange(n_sim)[:, None], a] -
                    dfn[np.arange(n_sim)[:, None], b] + et)
        lb = np.exp(atk[np.arange(n_sim)[:, None], b] -
                    dfn[np.arange(n_sim)[:, None], a] + et)
        rho_exp = np.repeat(
            rho[:, None], n, axis=1) if usa_dixon_coles else None

        ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
        vence_a = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))
        return np.where(vence_a, a, b)

    r32 = classificados
    contar_fase(r32, "round_of_32")

    r16 = rodada(r32)
    contar_fase(r16, "round_of_16")

    qf = rodada(r16)
    contar_fase(qf, "quarter_finalists")

    sf = rodada(qf)
    contar_fase(sf, "semi_finalists")

    fin = rodada(sf)
    contar_fase(fin, "finalists")

    champ = rodada(fin)
    contar_fase(champ, "champion")

    probs = {fase: stats[fase] / n_sim for fase in stats.keys()}
    return probs


def simular_copa_2022(post_draws, teams_list, grupos, n_sim=100000):
    atk_draws, dfn_draws, eta_draws = post_draws['attack'], post_draws['defense'], post_draws['eta']
    usa_dixon_coles = 'rho' in post_draws
    rho_draws = post_draws['rho'] if usa_dixon_coles else None

    n_samples, n_teams = len(eta_draws), len(teams_list)
    t_idx = {name: i for i, name in enumerate(teams_list)}
    g_indices = np.array([[t_idx[t] for t in ts] for ts in grupos.values()])

    idx_samples = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[idx_samples], dfn_draws[idx_samples]
    et = eta_draws[idx_samples].reshape(-1, 1)
    rho = rho_draws[idx_samples] if usa_dixon_coles else None

    # Estatísticas monitoradas para 2022
    stats = {f: np.zeros(n_teams) for f in [
        "avancou_grupos", "quarter_finalists", "semi_finalists", "finalists", "champion"]}

    # --- FASE DE GRUPOS (8 Grupos) ---
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pts, sg, gp = np.zeros((n_sim, 8, 4)), np.zeros(
        (n_sim, 8, 4)), np.zeros((n_sim, 8, 4))

    for p1, p2 in pairs:
        i1, i2 = g_indices[:, p1], g_indices[:, p2]
        l1 = np.exp(atk[:, i1] - dfn[:, i2] + et)
        l2 = np.exp(atk[:, i2] - dfn[:, i1] + et)

        rho_exp = np.repeat(
            rho[:, None], 8, axis=1) if usa_dixon_coles else None

        # Chama a função Dixon-Coles SUPER RÁPIDA
        g1, g2 = simular_jogos(l1, l2, rho_exp, n_sim)

        pts[:, :, p1] += (g1 > g2)*3 + (g1 == g2)
        pts[:, :, p2] += (g2 > g1)*3 + (g1 == g2)
        sg[:, :, p1] += (g1 - g2)
        sg[:, :, p2] += (g2 - g1)
        gp[:, :, p1] += g1
        gp[:, :, p2] += g2

    sort_val = pts * 1000000 + sg * 1000 + gp
    ranks = np.argsort(-sort_val, axis=2)

    # Seleciona 1º (c1) e 2º (c2) de cada um dos 8 grupos instantaneamente
    c1 = np.array([g_indices[g, ranks[:, g, 0]] for g in range(8)]).T
    c2 = np.array([g_indices[g, ranks[:, g, 1]] for g in range(8)]).T

    # FIM DO FOR LENTO: Função contadora instantânea do NumPy
    def contar_fase(array_fase, nome_fase):
        unique, counts = np.unique(array_fase, return_counts=True)
        stats[nome_fase][unique] += counts

    contar_fase(c1, "avancou_grupos")
    contar_fase(c2, "avancou_grupos")

    # --- MATA-MATA ---
    def rodada(ta, tb):
        n_matches = ta.shape[1]
        la = np.exp(atk[np.arange(n_sim)[:, None], ta] -
                    dfn[np.arange(n_sim)[:, None], tb] + et)
        lb = np.exp(atk[np.arange(n_sim)[:, None], tb] -
                    dfn[np.arange(n_sim)[:, None], ta] + et)

        rho_exp = np.repeat(rho[:, None], n_matches,
                            axis=1) if usa_dixon_coles else None
        ga, gb = simular_jogos(la, lb, rho_exp, n_sim)

        vence_a = (ga > gb) | ((ga == gb) & (
            np.random.rand(n_sim, n_matches) < 0.5))
        return np.where(vence_a, ta, tb)

    # Chaveamento oficial das Oitavas de 2022
    oitavas_a = np.column_stack(
        [c1[:, 0], c1[:, 2], c1[:, 4], c1[:, 6], c1[:, 1], c1[:, 3], c1[:, 5], c1[:, 7]])
    oitavas_b = np.column_stack(
        [c2[:, 1], c2[:, 3], c2[:, 5], c2[:, 7], c2[:, 0], c2[:, 2], c2[:, 4], c2[:, 6]])

    quartas = rodada(oitavas_a, oitavas_b)
    contar_fase(quartas, "quarter_finalists")

    semis = rodada(quartas[:, 0::2], quartas[:, 1::2])
    contar_fase(semis, "semi_finalists")

    finalistas = rodada(semis[:, 0::2], semis[:, 1::2])
    contar_fase(finalistas, "finalists")

    campeao = rodada(finalistas[:, [0]], finalistas[:, [1]])
    contar_fase(campeao, "champion")

    probs = {fase: stats[fase] / n_sim for fase in stats.keys()}
    return probs
