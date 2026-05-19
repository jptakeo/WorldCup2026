import os

import numpy as np
import pandas as pd


def filter_low_volume_teams(
    df, min_matches=20, col_home="home_team", col_away="away_team"
):
    """Keep only matches where both teams have enough historical volume."""
    home_matches = df[col_home].value_counts()
    away_matches = df[col_away].value_counts()

    # Combine home and away appearances; fill missing sides for one-sided histories.
    total_matches = home_matches.add(away_matches, fill_value=0)
    valid_teams = total_matches[total_matches > min_matches].index

    clean_df = df[
        df[col_home].isin(valid_teams) & df[col_away].isin(valid_teams)
    ].copy()

    # Keep Stan team IDs deterministic across runs.
    team_list = sorted(list(valid_teams))

    return clean_df, team_list


def get_importance_factor(row):
    """Calculate match importance from the tournament name."""
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


def prepare_cycle_data(
    csv_path,
    start_date,
    end_date=None,
    min_matches=20,
    apply_decay=True,
    decay_alpha=0.001,
):
    """
    Filter matches for a specific cycle, apply weights, and map teams.

    This supports the 2026 cycle and historical validation cycles.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado.")

    df = pd.read_csv(csv_path)
    df, _ = filter_low_volume_teams(df)
    df["date"] = pd.to_datetime(df["date"])

    # Limit training matches to the requested World Cup cycle.
    date_filter = df["date"] >= start_date
    if end_date:
        date_filter &= df["date"] < end_date
    df_cycle = df[date_filter].copy()

    # Re-check minimum volume after the cycle filter.
    counts = (
        df_cycle["home_team"]
        .value_counts()
        .add(df_cycle["away_team"].value_counts(), fill_value=0)
    )
    valid_cycle_teams = counts[counts >= min_matches].index
    df_cycle = df_cycle[
        df_cycle["home_team"].isin(valid_cycle_teams)
        & df_cycle["away_team"].isin(valid_cycle_teams)
    ].copy()

    df_cycle["tourn_weight"] = df_cycle.apply(get_importance_factor, axis=1)

    # Weight recent and high-importance matches more heavily.
    if apply_decay:
        max_date = df_cycle["date"].max()
        df_cycle["days_ago"] = (max_date - df_cycle["date"]).dt.days
        df_cycle["time_weight"] = np.exp(-decay_alpha * df_cycle["days_ago"])
        df_cycle["game_weight"] = (df_cycle["tourn_weight"] * df_cycle["time_weight"])
    else:
        df_cycle["game_weight"] = df_cycle["tourn_weight"]
    
    df_cycle['game_weight'] = df_cycle['game_weight'] / 10

    # Stan uses 1-based team IDs.
    teams_list = sorted(list(valid_cycle_teams))
    team_map = {name: i + 1 for i, name in enumerate(teams_list)}

    print(
        f"Dados preparados: {len(df_cycle)} jogos e {len(teams_list)} seleções identificadas."
    )

    return df_cycle, teams_list, team_map


def load_ranking_priors(ranking_path, teams_list):
    """
    Read the FIFA ranking CSV and return normalized strengths ordered by teams_list.
    """
    try:
        df_ranking = pd.read_csv(ranking_path, sep=None, engine="python")
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

    print(f"Sucesso! Posições do ranking {ranking_path} carregadas e normalizadas.")
    return np.array(team_prior_strength)
