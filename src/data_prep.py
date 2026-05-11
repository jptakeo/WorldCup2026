import os

import numpy as np
import pandas as pd


def remover_selecoes_sem_volume(
    df, min_jogos=20, col_home="home_team", col_away="away_team"
):
    """Keep only matches where both teams have enough historical volume."""
    jogos_casa = df[col_home].value_counts()
    jogos_fora = df[col_away].value_counts()

    # Combine home and away appearances; fill missing sides for one-sided histories.
    total_jogos = jogos_casa.add(jogos_fora, fill_value=0)
    selecoes_validas = total_jogos[total_jogos > min_jogos].index

    df_limpo = df[
        df[col_home].isin(selecoes_validas) & df[col_away].isin(selecoes_validas)
    ].copy()

    # Keep Stan team IDs deterministic across runs.
    lista_selecoes = sorted(list(selecoes_validas))

    return df_limpo, lista_selecoes


def obter_fator_i(row):
    """Calcula a importância da partida com base no torneio."""
    t = str(row["tournament"]).lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    continental_comps = [
        "uefa euro",
        "copa américa",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "oceania nations cup",
    ]
    if any(comp in t for comp in continental_comps) and "qualification" not in t:
        return 40
    if "nations league" in t and "final" in t:
        return 35
    if "qualification" in t or "eliminatórias" in t:
        return 25
    if "nations league" in t:
        return 20
    if "friendly" in t:
        return 10
    return 20


def preparar_dados_ciclo(
    caminho_csv,
    data_inicio,
    data_fim=None,
    min_jogos=20,
    aplicar_decaimento=True,
    decay_alpha=0.001,
):
    """
    Filtra os jogos para um ciclo específico, aplica pesos e mapeia os times.
    Serve tanto para o ciclo de 2026 quanto para o de 2022.
    """
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo {caminho_csv} não encontrado.")

    df = pd.read_csv(caminho_csv)
    df, _ = remover_selecoes_sem_volume(df)
    df["date"] = pd.to_datetime(df["date"])

    # Limit training matches to the requested World Cup cycle.
    filtro = df["date"] >= data_inicio
    if data_fim:
        filtro &= df["date"] < data_fim
    df_cycle = df[filtro].copy()

    # Re-check minimum volume after the cycle filter.
    counts = (
        df_cycle["home_team"]
        .value_counts()
        .add(df_cycle["away_team"].value_counts(), fill_value=0)
    )
    teams_validos = counts[counts >= min_jogos].index
    df_cycle = df_cycle[
        df_cycle["home_team"].isin(teams_validos)
        & df_cycle["away_team"].isin(teams_validos)
    ].copy()

    df_cycle["tourn_weight"] = df_cycle.apply(obter_fator_i, axis=1)

    # Weight recent and high-importance matches more heavily.
    if aplicar_decaimento:
        max_date = df_cycle["date"].max()
        df_cycle["days_ago"] = (max_date - df_cycle["date"]).dt.days
        df_cycle["time_weight"] = np.exp(-decay_alpha * df_cycle["days_ago"])
        df_cycle["game_weight"] = (df_cycle["tourn_weight"] * df_cycle["time_weight"]) / 10.0
    else:
        df_cycle["game_weight"] = df_cycle["fator_i"] / 10.0

    # Stan uses 1-based team IDs.
    teams_list = sorted(list(teams_validos))
    team_map = {name: i + 1 for i, name in enumerate(teams_list)}

    print(
        f"Dados preparados: {len(df_cycle)} jogos e {len(teams_list)} seleções identificadas."
    )

    return df_cycle, teams_list, team_map


def carregar_priors_ranking(caminho_ranking, teams_list):
    """
    Lê o CSV de ranking da FIFA, normaliza os pontos (z-score)
    e retorna o vetor ordenado de acordo com o teams_list.
    """
    try:
        df_ranking = pd.read_csv(caminho_ranking, sep=None, engine="python")
    except Exception as e:
        print(f"Erro ao ler o arquivo de ranking: {e}")
        return np.zeros(len(teams_list))

    cols = df_ranking.columns.tolist()
    team_col = next(
        (c for c in cols if "sele" in c.lower() or "team" in c.lower()),
        cols[min(1, len(cols) - 1)],
    )
    pts_col = next(
        (
            c
            for c in cols
            if "pont" in c.lower() or "pts" in c.lower() or "point" in c.lower()
        ),
        cols[-1],
    )

    fifa_ranking_data = pd.Series(
        df_ranking[pts_col].values, index=df_ranking[team_col]
    ).to_dict()
    all_pts = [
        float(str(v).replace(",", "."))
        for v in fifa_ranking_data.values()
        if pd.notnull(v)
    ]

    mean_pts = np.mean(all_pts)
    std_pts = np.std(all_pts)

    team_prior_strength = []
    for team in teams_list:
        # Missing teams receive the mean ranking score before normalization.
        pts = fifa_ranking_data.get(team, mean_pts)
        normalized_val = (float(str(pts).replace(",", ".")) - mean_pts) / std_pts
        team_prior_strength.append(normalized_val)

    print(f"Sucesso! Posições do ranking {caminho_ranking} carregadas e normalizadas.")
    return np.array(team_prior_strength)
