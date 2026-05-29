from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    ALL_MATCHUPS_EXPORT_COLS,
    PARTIDAS_EXPORT_COLS,
    TEAM_MAP_EN_TO_PT,
    TEAM_MAP_PT_TO_EN,
)

# Precomputed factorials for score probabilities from 0 to 10 goals.
FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800])

NUM_TO_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}

DEFAULT_DRAWS_PATH = "data/outputs/models/draws_2026_n_poisson_ranking.npz"
DEFAULT_MATCH_SIMS = 100_000
ALL_MATCHUPS_BATCH_SIZE = 128

current_results_df = pd.read_csv("data/world_cup_results.csv")


def load_draws(path: str | Path) -> dict[str, np.ndarray]:
    """Load Stan posterior draws saved as ``.npz``."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


def _load_schedule_orientations(
    results_path: str | Path | pd.DataFrame = "data/world_cup_results.csv",
) -> dict[frozenset[str], tuple[str, str, str, object]]:
    """Map unordered PT pair -> (home_pt, away_pt, group, date)."""
    if isinstance(results_path, pd.DataFrame):
        results_df = results_path.copy()
    else:
        results_df = pd.read_csv(results_path)
    for col in ["home_team", "away_team", "group"]:
        results_df[col] = results_df[col].astype(str).str.strip()

    orientations: dict[frozenset[str], tuple[str, str, str, object]] = {}
    for row in results_df.itertuples(index=False):
        orientations[frozenset({row.home_team, row.away_team})] = (
            row.home_team,
            row.away_team,
            row.group,
            row.date,
        )
    return orientations


def _resolve_fixture_orientation(
    team_a_en: str,
    team_b_en: str,
    schedule: dict[frozenset[str], tuple[str, str, str, object]],
) -> tuple[str, str, str | None, object | None]:
    team_a_pt = TEAM_MAP_EN_TO_PT.get(team_a_en, team_a_en)
    team_b_pt = TEAM_MAP_EN_TO_PT.get(team_b_en, team_b_en)
    scheduled = schedule.get(frozenset({team_a_pt, team_b_pt}))

    if scheduled is None:
        return team_a_en, team_b_en, None, None

    home_pt, away_pt, group, date = scheduled
    home_en = TEAM_MAP_PT_TO_EN.get(home_pt, home_pt)
    away_en = TEAM_MAP_PT_TO_EN.get(away_pt, away_pt)
    return home_en, away_en, group, date


def _sample_posterior(
    post_draws: dict[str, np.ndarray],
    n_sim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    n_samples = len(post_draws["attack"])
    sample_idx = np.random.choice(n_samples, n_sim)
    atk = post_draws["attack"][sample_idx]
    dfn = post_draws["defense"][sample_idx]
    rho = post_draws["rho"][sample_idx] if "rho" in post_draws else None
    et = post_draws["eta"][sample_idx]
    return atk, dfn, rho, et


def _match_lambdas(
    atk: np.ndarray,
    dfn: np.ndarray,
    home_idx: int | np.ndarray,
    away_idx: int | np.ndarray,
    et: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson intensities with Stan ``eta`` added on both sides (log-scale)."""
    et = et.reshape(-1, 1)
    n_sim = atk.shape[0]
    if np.isscalar(home_idx) or (
        isinstance(home_idx, np.ndarray) and home_idx.ndim == 1
    ):
        return (
            np.exp(atk[:, home_idx] - dfn[:, away_idx] + et),
            np.exp(atk[:, away_idx] - dfn[:, home_idx] + et),
        )
    row_idx = np.arange(n_sim)[:, None]
    return (
        np.exp(atk[row_idx, home_idx] - dfn[row_idx, away_idx] + et),
        np.exp(atk[row_idx, away_idx] - dfn[row_idx, home_idx] + et),
    )


def _aggregate_match_probs(g1: np.ndarray, g2: np.ndarray) -> dict[str, float]:
    """Empirical outcome and scoreline probabilities (%) from simulated goals."""
    row: dict[str, float] = {
        "home_win": float(np.mean(g1 > g2) * 100),
        "draw": float(np.mean(g1 == g2) * 100),
        "away_win": float(np.mean(g1 < g2) * 100),
    }
    for i in range(5):
        for j in range(5):
            key = f"{NUM_TO_WORD[i]}_{NUM_TO_WORD[j]}"
            row[key] = float(np.mean((g1 == i) & (g2 == j)) * 100)
    return row


def _simulate_match_goals(
    atk: np.ndarray,
    dfn: np.ndarray,
    rho: np.ndarray | None,
    et: np.ndarray,
    home_idx: int | np.ndarray,
    away_idx: int | np.ndarray,
    n_sim: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_pairs = 1 if np.isscalar(home_idx) else len(home_idx)
    l1, l2 = _match_lambdas(atk, dfn, home_idx, away_idx, et)
    rho_exp = None
    if rho is not None:
        rho_exp = np.repeat(rho[:, None], n_pairs, axis=1)
    return simulate_matches(l1, l2, rho_exp, n_sim)


def build_all_matchups_dataframe_mc(
    teams_list: list[str],
    wc_teams: list[str],
    atk: np.ndarray,
    dfn: np.ndarray,
    rho: np.ndarray | None,
    et: np.ndarray,
    n_sim: int,
    pair_goals_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    | None = None,
    schedule_path: str | Path = "data/world_cup_results.csv",
    batch_size: int = ALL_MATCHUPS_BATCH_SIZE,
) -> pd.DataFrame:
    """Monte Carlo match probabilities for every WC team pair.

    Uses Stan ``eta`` on both sides.
    """
    t_to_idx = {name: i for i, name in enumerate(teams_list)}
    schedule = _load_schedule_orientations(schedule_path)
    cache = pair_goals_cache or {}

    pairs: list[tuple[str, str]] = list(combinations(wc_teams, 2))
    rows: list[dict] = []

    pending: list[tuple[str, str, str, str, int]] = []

    def flush_pending() -> None:
        if not pending:
            return
        home_idxs_arr = np.array([item[4] for item in pending], dtype=int)
        away_idxs_arr = np.array([item[5] for item in pending], dtype=int)
        g1, g2 = _simulate_match_goals(
            atk, dfn, rho, et, home_idxs_arr, away_idxs_arr, n_sim
        )
        for m, (home_pt, away_pt, _home_en, _away_en, _, _) in enumerate(pending):
            rows.append(
                {
                    "home_team": home_pt,
                    "away_team": away_pt,
                    **_aggregate_match_probs(g1[:, m], g2[:, m]),
                }
            )
        pending.clear()

    for team_a_en, team_b_en in pairs:
        home_en, away_en, _, _ = _resolve_fixture_orientation(
            team_a_en, team_b_en, schedule
        )
        home_pt = TEAM_MAP_EN_TO_PT.get(home_en, home_en)
        away_pt = TEAM_MAP_EN_TO_PT.get(away_en, away_en)
        cache_key = (home_en, away_en)

        if cache_key in cache:
            hg, ag = cache[cache_key]
            rows.append(
                {
                    "home_team": home_pt,
                    "away_team": away_pt,
                    **_aggregate_match_probs(hg, ag),
                }
            )
            continue

        pending.append(
            (
                home_pt,
                away_pt,
                home_en,
                away_en,
                t_to_idx[home_en],
                t_to_idx[away_en],
            )
        )
        if len(pending) >= batch_size:
            flush_pending()

    flush_pending()

    df = pd.DataFrame(rows).round(4)
    return df[ALL_MATCHUPS_EXPORT_COLS]


def export_stan_match_csvs(
    teams_list: list[str],
    wc_teams: list[str],
    partidas_df: pd.DataFrame,
    atk: np.ndarray,
    dfn: np.ndarray,
    rho: np.ndarray | None,
    et: np.ndarray,
    n_sim: int,
    pair_goals_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    | None = None,
    partidas_path: str | Path = "docs/csv/previsoes/partidas.csv",
    all_matchups_path: str | Path = "docs/csv/previsoes/all_matchups.csv",
) -> None:
    """Write ``partidas.csv`` and ``all_matchups.csv`` from Stan Monte Carlo draws."""
    partidas_out = partidas_df[PARTIDAS_EXPORT_COLS].round(4)
    partidas_out.to_csv(partidas_path, index=False)
    partidas_out.to_csv("data/probs_fase_de_grupos.csv", index=False)

    print("Gerando all_matchups.csv (Monte Carlo Stan)...")
    all_df = build_all_matchups_dataframe_mc(
        teams_list,
        wc_teams,
        atk,
        dfn,
        rho,
        et,
        n_sim=n_sim,
        pair_goals_cache=pair_goals_cache,
    )
    all_df.to_csv(all_matchups_path, index=False)
    print(f"  Salvo em: {all_matchups_path} ({len(all_df)} confrontos)")


def simulate_matches(mu1, mu2, rho_draws=None, n_sim=100000, max_goals=10):
    if rho_draws is None:
        # Independent Poisson path when no Dixon-Coles rho is available.
        return np.random.poisson(mu1), np.random.poisson(mu2)

    if mu1.ndim == 1:
        mu1, mu2, rho_draws = mu1[:, None], mu2[:, None], rho_draws[:, None]

    rho = rho_draws
    n_matches = mu1.shape[1]
    goals = np.arange(max_goals + 1)

    # Vectorized Poisson PMF; avoids scipy overhead in the simulation loop.
    mu1_exp = mu1[:, :, None, None]
    mu2_exp = mu2[:, :, None, None]

    p1 = (
        np.exp(-mu1_exp)
        * (mu1_exp ** goals[None, None, :, None])
        / FACTORIALS[None, None, :, None]
    )
    p2 = (
        np.exp(-mu2_exp)
        * (mu2_exp ** goals[None, None, None, :])
        / FACTORIALS[None, None, None, :]
    )

    p_matrix = p1 * p2

    # Dixon-Coles low-score correction.
    rho_exp = rho[:, :, None, None]
    p_matrix[:, :, 0, 0] *= (1 - mu1_exp * mu2_exp * rho_exp)[:, :, 0, 0]
    p_matrix[:, :, 1, 0] *= (1 + mu2_exp * rho_exp)[:, :, 0, 0]
    p_matrix[:, :, 0, 1] *= (1 + mu1_exp * rho_exp)[:, :, 0, 0]
    p_matrix[:, :, 1, 1] *= (1 - rho_exp)[:, :, 0, 0]

    p_matrix = np.clip(p_matrix, a_min=0, a_max=None)

    # Sample one scoreline from each flattened cumulative distribution.
    p_flat = p_matrix.reshape(n_sim, n_matches, -1)
    p_flat /= p_flat.sum(axis=2, keepdims=True)

    cum_p = np.cumsum(p_flat, axis=2)
    rand_vals = np.random.rand(n_sim, n_matches, 1)

    score_idx = np.argmax(cum_p > rand_vals, axis=2)

    g1 = score_idx // (max_goals + 1)
    g2 = score_idx % (max_goals + 1)

    return g1, g2


def simulate_world_cup_2022(post_draws, teams_list, groups, n_sim=100000):
    atk_draws, dfn_draws, eta_draws = (
        post_draws["attack"],
        post_draws["defense"],
        post_draws["eta"],
    )
    uses_dixon_coles = "rho" in post_draws
    rho_draws = post_draws["rho"] if uses_dixon_coles else None

    n_samples, n_teams = len(eta_draws), len(teams_list)
    t_idx = {name: i for i, name in enumerate(teams_list)}
    g_indices = np.array([[t_idx[t] for t in ts] for ts in groups.values()])

    sample_idx = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[sample_idx], dfn_draws[sample_idx]
    et = eta_draws[sample_idx].reshape(-1, 1)
    rho = rho_draws[sample_idx] if uses_dixon_coles else None

    # Stage counters for the 32-team historical tournament format.
    stats = {
        f: np.zeros(n_teams)
        for f in [
            "avancou_grupos",
            "quarter_finalists",
            "semi_finalists",
            "finalists",
            "champion",
        ]
    }

    # Group stage: all six pairings per four-team group.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pts, sg, gp = (
        np.zeros((n_sim, 8, 4)),
        np.zeros((n_sim, 8, 4)),
        np.zeros((n_sim, 8, 4)),
    )

    for p1, p2 in pairs:
        i1, i2 = g_indices[:, p1], g_indices[:, p2]
        l1 = np.exp(atk[:, i1] - dfn[:, i2] + et)
        l2 = np.exp(atk[:, i2] - dfn[:, i1] + et)

        rho_exp = np.repeat(rho[:, None], 8, axis=1) if uses_dixon_coles else None

        g1, g2 = simulate_matches(l1, l2, rho_exp, n_sim)

        pts[:, :, p1] += (g1 > g2) * 3 + (g1 == g2)
        pts[:, :, p2] += (g2 > g1) * 3 + (g1 == g2)
        sg[:, :, p1] += g1 - g2
        sg[:, :, p2] += g2 - g1
        gp[:, :, p1] += g1
        gp[:, :, p2] += g2

    sort_val = pts * 1000000 + sg * 1000 + gp
    ranks = np.argsort(-sort_val, axis=2)

    # Select each group's winner and runner-up after ranking by points, goal
    # difference, then goals scored.
    c1 = np.array([g_indices[g, ranks[:, g, 0]] for g in range(8)]).T
    c2 = np.array([g_indices[g, ranks[:, g, 1]] for g in range(8)]).T

    def count_stage(stage_array, stage_name):
        unique, counts = np.unique(stage_array, return_counts=True)
        stats[stage_name][unique] += counts

    count_stage(c1, "avancou_grupos")
    count_stage(c2, "avancou_grupos")

    # Knockout helper: ties go to penalties with a 50/50 split.
    def play_round(ta, tb):
        n_matches = ta.shape[1]
        la = np.exp(
            atk[np.arange(n_sim)[:, None], ta] - dfn[np.arange(n_sim)[:, None], tb] + et
        )
        lb = np.exp(
            atk[np.arange(n_sim)[:, None], tb] - dfn[np.arange(n_sim)[:, None], ta] + et
        )

        rho_exp = (
            np.repeat(rho[:, None], n_matches, axis=1) if uses_dixon_coles else None
        )
        ga, gb = simulate_matches(la, lb, rho_exp, n_sim)

        team_a_wins = (ga > gb) | (
            (ga == gb) & (np.random.rand(n_sim, n_matches) < 0.5)
        )
        return np.where(team_a_wins, ta, tb)

    # Round-of-16 bracket ordering for the 2018/2022 format.
    first_place_idx = [0, 2, 4, 6, 1, 3, 5, 7]  # 1A, 1C, 1E, 1G, 1B, 1D, 1F, 1H
    second_place_idx = [1, 3, 5, 7, 0, 2, 4, 6]  # 2B, 2D, 2F, 2H, 2A, 2C, 2E, 2G

    round16_side_a = c1[:, first_place_idx]
    round16_side_b = c2[:, second_place_idx]

    quarterfinalists = play_round(round16_side_a, round16_side_b)
    count_stage(quarterfinalists, "quarter_finalists")

    # Winners are paired by adjacent bracket slots in each following round.
    semifinalists = play_round(quarterfinalists[:, 0::2], quarterfinalists[:, 1::2])
    count_stage(semifinalists, "semi_finalists")

    finalists = play_round(semifinalists[:, 0::2], semifinalists[:, 1::2])
    count_stage(finalists, "finalists")

    champion = play_round(finalists[:, [0]], finalists[:, [1]])
    count_stage(champion, "champion")

    probs = {stage: stats[stage] / n_sim for stage in stats}
    return probs


def simulate_stage_and_remaining(
    post_draws, teams_list, matches_df, stage_names=None, n_sim=100_000
):
    """
    Simula uma fase específica e todas as subsequentes até o campeão.
    O matches_df deve estar ordenado em formato de chaveamento
    (0 enfrenta 1, 2 enfrenta 3...).
    """
    atk_draws, dfn_draws, eta_draws = (
        post_draws["attack"],
        post_draws["defense"],
        post_draws["eta"],
    )
    uses_dixon_coles = "rho" in post_draws
    rho_draws = post_draws["rho"] if uses_dixon_coles else None

    n_samples, n_teams_total = len(eta_draws), len(teams_list)
    t_to_idx = {name: i for i, name in enumerate(teams_list)}

    sample_idx = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[sample_idx], dfn_draws[sample_idx]
    et = eta_draws[sample_idx].reshape(-1, 1)
    rho = rho_draws[sample_idx] if uses_dixon_coles else None

    num_matches = len(matches_df)

    # Infer the phase labels from the size of the first knockout round.
    if stage_names is None:
        stage_mapping = {
            16: [
                "round_of_32",
                "round_of_16",
                "quarter_finalists",
                "semi_finalists",
                "finalists",
                "champion",
            ],
            8: [
                "round_of_16",
                "quarter_finalists",
                "semi_finalists",
                "finalists",
                "champion",
            ],
            4: ["quarter_finalists", "semi_finalists", "finalists", "champion"],
            2: ["semi_finalists", "finalists", "champion"],
            1: ["finalists", "champion"],
        }
    stage_names = stage_mapping.get(
        num_matches,
        ["fase_atual"]
        + [f"avanco_{i}" for i in range(1, int(np.log2(num_matches)) + 2)],
    )

    stats = {f: np.zeros(n_teams_total) for f in stage_names}
    match_stats = []
    num_to_word = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}

    # Competitor matrix tracks the surviving team ID in each bracket slot.
    competitors = np.zeros((n_sim, num_matches * 2), dtype=int)

    involved_teams = []
    for m in range(num_matches):
        row = matches_df.iloc[m]
        t1, t2 = row["home_team"], row["away_team"]
        involved_teams.extend([t1, t2])
        i1, i2 = t_to_idx[t1], t_to_idx[t2]

        competitors[:, m * 2] = i1
        competitors[:, m * 2 + 1] = i2

        # Teams in the provided fixture list are guaranteed to be in the initial phase.
        stats[stage_names[0]][i1] = n_sim
        stats[stage_names[0]][i2] = n_sim

    stage_idx = 1
    is_first_round = True

    # Simulate each round in cascade until one champion remains.
    while competitors.shape[1] > 1:
        n = competitors.shape[1] // 2
        a, b = competitors[:, 0::2], competitors[:, 1::2]

        la = np.exp(
            atk[np.arange(n_sim)[:, None], a] - dfn[np.arange(n_sim)[:, None], b] + et
        )
        lb = np.exp(
            atk[np.arange(n_sim)[:, None], b] - dfn[np.arange(n_sim)[:, None], a] + et
        )
        rho_exp = np.repeat(rho[:, None], n, axis=1) if uses_dixon_coles else None

        ga, gb = simulate_matches(la, lb, rho_exp, n_sim)

        team_a_wins = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))

        # Export match-level probabilities only for the phase supplied by matches_df.
        if is_first_round:
            for m in range(num_matches):
                g1, g2 = ga[:, m], gb[:, m]

                match_info = {
                    "home_team": matches_df.iloc[m]["home_team"],
                    "away_team": matches_df.iloc[m]["away_team"],
                    "date": matches_df.iloc[m].get("date", "Data não encontrada"),
                    "home_win": np.mean(g1 > g2) * 100,
                    "draw": np.mean(g1 == g2) * 100,
                    "away_win": np.mean(g1 < g2) * 100,
                }

                for i in range(5):
                    for j in range(5):
                        col_name = f"{num_to_word[i]}_{num_to_word[j]}"
                        match_info[col_name] = np.mean((g1 == i) & (g2 == j)) * 100

                match_stats.append(match_info)
            is_first_round = False

        competitors = np.where(team_a_wins, a, b)

        # Count teams that reached the next phase.
        if stage_idx < len(stage_names):
            unique, counts = np.unique(competitors, return_counts=True)
            stats[stage_names[stage_idx]][unique] += counts

        stage_idx += 1

    # Build stage and match exports expected by downstream scripts.
    probs = {stage: stats[stage] / n_sim for stage in stage_names}

    df_summary = pd.DataFrame(stats)
    df_summary = (df_summary / n_sim * 100).round(2)
    df_summary.insert(0, "team", teams_list)

    # Keep only teams involved in the input phase.
    df_summary = df_summary[df_summary["team"].isin(involved_teams)]

    # Rank by the deepest computed phase, usually champion probability.
    sort_col = stage_names[-1]
    df_summary = df_summary.sort_values(by=sort_col, ascending=False).reset_index(
        drop=True
    )
    df_summary.insert(0, "position", df_summary.index + 1)

    df_csv = pd.DataFrame(
        {
            "position": df_summary["position"],
            "team": df_summary["team"],
            "champion": df_summary.get(
                "champion", pd.Series([100.0] * len(df_summary))
            ),
            "final": df_summary.get("finalists", pd.Series([100.0] * len(df_summary))),
            "semifinals": df_summary.get(
                "semi_finalists", pd.Series([100.0] * len(df_summary))
            ),
            "quarterfinals": df_summary.get(
                "quarter_finalists", pd.Series([100.0] * len(df_summary))
            ),
            "round_of_16": df_summary.get(
                "round_of_16", pd.Series([100.0] * len(df_summary))
            ),
            "round_of_32": df_summary.get(
                "round_of_32", pd.Series([100.0] * len(df_summary))
            ),
        }
    )
    df_csv["team"] = df_csv["team"].replace(TEAM_MAP_EN_TO_PT)

    df_matches = pd.DataFrame(match_stats).round(2)
    df_matches.reset_index(drop=True, inplace=True)
    df_matches["home_team"] = df_matches["home_team"].replace(TEAM_MAP_EN_TO_PT)
    df_matches["away_team"] = df_matches["away_team"].replace(TEAM_MAP_EN_TO_PT)

    return probs, df_csv, df_matches


def simulate_world_cup_2026(
    post_draws,
    teams_list,
    groups,
    df_schedule=current_results_df,
    team_map_pt_to_en=TEAM_MAP_PT_TO_EN,
    team_map_en_to_pt=TEAM_MAP_EN_TO_PT,
    n_sim=100_000,
    seed=None,
):
    uses_dixon_coles = "rho" in post_draws
    atk, dfn, rho, et = _sample_posterior(post_draws, n_sim, seed=seed)

    n_teams = len(teams_list)

    # Map model team names to integer IDs for vectorized indexing.
    t_to_idx = {name: i for i, name in enumerate(teams_list)}
    g_indices = np.array([[t_to_idx[t] for t in ts] for ts in groups.values()])
    schedule = _load_schedule_orientations(df_schedule)

    original_stages = [
        "avancou_grupos",
        "round_of_32",
        "round_of_16",
        "quarter_finalists",
        "semi_finalists",
        "finalists",
        "champion",
    ]
    all_stages = original_stages + [
        "group_first_place",
        "group_second_place",
        "group_third_place",
    ]

    stats = {f: np.zeros(n_teams) for f in all_stages}

    # Attach model IDs to the public schedule so exported match dates stay aligned.
    name_to_idx = dict(t_to_idx)

    df_schedule["home_id"] = (
        df_schedule["home_team"].map(team_map_pt_to_en).map(name_to_idx).astype("Int64")
    )
    df_schedule["away_id"] = (
        df_schedule["away_team"].map(team_map_pt_to_en).map(name_to_idx).astype("Int64")
    )

    date_map = {}
    for _, row in df_schedule.iterrows():
        if pd.notna(row["home_id"]) and pd.notna(row["away_id"]):
            match_key = frozenset([row["home_id"], row["away_id"]])
            date_map[match_key] = row["date"]

    # Group stage: simulate every six-match group schedule.
    match_stats = []
    pair_goals_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pts, sg, gp = (
        np.zeros((n_sim, 12, 4)),
        np.zeros((n_sim, 12, 4)),
        np.zeros((n_sim, 12, 4)),
    )

    for p1, p2 in pairs:
        i1, i2 = g_indices[:, p1], g_indices[:, p2]
        l1, l2 = _match_lambdas(atk, dfn, i1, i2, et)
        rho_exp = np.repeat(rho[:, None], 12, axis=1) if uses_dixon_coles else None

        g1, g2 = simulate_matches(l1, l2, rho_exp, n_sim)

        for g in range(12):
            t1_idx, t2_idx = g_indices[g, p1], g_indices[g, p2]
            team1, team2 = teams_list[t1_idx], teams_list[t2_idx]
            match_date = date_map.get(
                frozenset([t1_idx, t2_idx]), "Data não encontrada"
            )
            g1_g, g2_g = g1[:, g], g2[:, g]

            home_en, away_en, sched_group, sched_date = _resolve_fixture_orientation(
                team1, team2, schedule
            )
            if team1 == home_en:
                hg, ag = g1_g, g2_g
            else:
                hg, ag = g2_g, g1_g

            pair_goals_cache[(home_en, away_en)] = (hg, ag)

            match_info = {
                "group": sched_group or list(groups.keys())[g],
                "home_team": home_en,
                "away_team": away_en,
                "date": sched_date if sched_date is not None else match_date,
                **_aggregate_match_probs(hg, ag),
            }
            match_stats.append(match_info)

        pts[:, :, p1] += (g1 > g2) * 3 + (g1 == g2)
        pts[:, :, p2] += (g2 > g1) * 3 + (g1 == g2)
        sg[:, :, p1] += g1 - g2
        sg[:, :, p2] += g2 - g1
        gp[:, :, p1] += g1
        gp[:, :, p2] += g2

    sort_val = pts * 1000000 + sg * 1000 + gp
    ranks = np.argsort(-sort_val, axis=2)

    def count_stage(stage_array, stage_name):
        unique, counts = np.unique(stage_array, return_counts=True)
        stats[stage_name][unique] += counts

    # Extract group placements with vectorized ranking arrays.
    firsts = np.zeros((n_sim, 12), dtype=int)
    seconds = np.zeros((n_sim, 12), dtype=int)
    all_thirds = np.zeros((n_sim, 12), dtype=int)
    third_place_scores = np.zeros((n_sim, 12))

    for g in range(12):
        firsts[:, g] = g_indices[g, ranks[:, g, 0]]
        seconds[:, g] = g_indices[g, ranks[:, g, 1]]
        all_thirds[:, g] = g_indices[g, ranks[:, g, 2]]
        third_place_scores[:, g] = sort_val[np.arange(n_sim), g, ranks[:, g, 2]]

        count_stage(firsts[:, g], "group_first_place")
        count_stage(seconds[:, g], "group_second_place")
        count_stage(all_thirds[:, g], "group_third_place")

    # Select the eight best third-place teams in each simulation.
    best_3rd_group_idx = np.argsort(-third_place_scores, axis=1)[:, :8]
    sorted_thirds = np.sort(best_3rd_group_idx, axis=1)

    # Allowed third-place assignments by winning-group slot (A=0, ..., L=11).
    valid_groups_for_1st_idx = {
        4: [0, 1, 2, 3, 5],  # 1E
        8: [2, 3, 5, 6, 7],  # 1I
        3: [1, 4, 5, 8, 9],  # 1D
        6: [0, 4, 7, 8, 9],  # 1G
        0: [2, 4, 5, 7, 8],  # 1A
        11: [4, 7, 8, 9, 10],  # 1L
        1: [4, 5, 6, 8, 9],  # 1B
        10: [3, 4, 8, 9, 11],  # 1K
    }
    slots_idx = [4, 8, 3, 6, 0, 11, 1, 10]
    allocation_cache = {}

    def get_allocation(thirds_tuple):
        if thirds_tuple in allocation_cache:
            return allocation_cache[thirds_tuple]
        thirds_list = list(thirds_tuple)
        alloc = {}

        def backtrack(idx, available):
            if idx == 8:
                return True
            slot = slots_idx[idx]
            for t in available:
                if t in valid_groups_for_1st_idx[slot]:
                    alloc[slot] = t
                    new_avail = available.copy()
                    new_avail.remove(t)
                    if backtrack(idx + 1, new_avail):
                        return True
                    del alloc[slot]
            return False

        if backtrack(0, thirds_list):
            res = [alloc[s] for s in slots_idx]
        else:
            res = thirds_list.copy()  # Defensive fallback for unexpected allocations.

        allocation_cache[thirds_tuple] = res
        return res

    allocated_thirds = np.zeros((n_sim, 8), dtype=int)
    for i in range(n_sim):
        allocated_thirds[i] = get_allocation(tuple(sorted_thirds[i]))

    # Build the round-of-32 bracket as ordered team-ID slots.
    row_idx = np.arange(n_sim)
    r32 = np.zeros((n_sim, 32), dtype=int)

    # Left side of the bracket.
    r32[:, 0], r32[:, 1] = (
        firsts[:, 4],
        all_thirds[row_idx, allocated_thirds[:, 0]],
    )  # 1E vs 3...
    r32[:, 2], r32[:, 3] = (
        firsts[:, 8],
        all_thirds[row_idx, allocated_thirds[:, 1]],
    )  # 1I vs 3...
    r32[:, 4], r32[:, 5] = seconds[:, 0], seconds[:, 1]  # 2A vs 2B
    r32[:, 6], r32[:, 7] = firsts[:, 5], seconds[:, 2]  # 1F vs 2C
    r32[:, 8], r32[:, 9] = seconds[:, 10], seconds[:, 11]  # 2K vs 2L
    r32[:, 10], r32[:, 11] = firsts[:, 7], seconds[:, 9]  # 1H vs 2J
    r32[:, 12], r32[:, 13] = (
        firsts[:, 3],
        all_thirds[row_idx, allocated_thirds[:, 2]],
    )  # 1D vs 3...
    r32[:, 14], r32[:, 15] = (
        firsts[:, 6],
        all_thirds[row_idx, allocated_thirds[:, 3]],
    )  # 1G vs 3...

    # Right side of the bracket.
    r32[:, 16], r32[:, 17] = firsts[:, 2], seconds[:, 5]  # 1C vs 2F
    r32[:, 18], r32[:, 19] = seconds[:, 4], seconds[:, 8]  # 2E vs 2I
    r32[:, 20], r32[:, 21] = (
        firsts[:, 0],
        all_thirds[row_idx, allocated_thirds[:, 4]],
    )  # 1A vs 3...
    r32[:, 22], r32[:, 23] = (
        firsts[:, 11],
        all_thirds[row_idx, allocated_thirds[:, 5]],
    )  # 1L vs 3...
    r32[:, 24], r32[:, 25] = firsts[:, 9], seconds[:, 7]  # 1J vs 2H
    r32[:, 26], r32[:, 27] = seconds[:, 3], seconds[:, 6]  # 2D vs 2G
    r32[:, 28], r32[:, 29] = (
        firsts[:, 1],
        all_thirds[row_idx, allocated_thirds[:, 6]],
    )  # 1B vs 3...
    r32[:, 30], r32[:, 31] = (
        firsts[:, 10],
        all_thirds[row_idx, allocated_thirds[:, 7]],
    )  # 1K vs 3...

    count_stage(r32, "avancou_grupos")
    count_stage(r32, "round_of_32")

    # Knockout rounds for the 2026 bracket.
    def play_round(competitors):
        n = competitors.shape[1] // 2
        a, b = competitors[:, 0::2], competitors[:, 1::2]
        la, lb = _match_lambdas(atk, dfn, a, b, et)
        rho_exp = np.repeat(rho[:, None], n, axis=1) if uses_dixon_coles else None

        ga, gb = simulate_matches(la, lb, rho_exp, n_sim)
        team_a_wins = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))
        return np.where(team_a_wins, a, b)

    r16 = play_round(r32)
    count_stage(r16, "round_of_16")

    qf = play_round(r16)
    count_stage(qf, "quarter_finalists")

    sf = play_round(qf)
    count_stage(sf, "semi_finalists")

    fin = play_round(sf)
    count_stage(fin, "finalists")

    champ = play_round(fin)
    count_stage(champ, "champion")

    # Export stage summary and group-match probability tables.
    df_summary = pd.DataFrame(stats)
    df_summary = (df_summary / n_sim * 100).round(2)
    df_summary.insert(0, "team", teams_list)

    teams_in_groups = [
        time for team_group_list in groups.values() for time in team_group_list
    ]
    df_summary = df_summary[df_summary["team"].isin(teams_in_groups)]
    df_summary = df_summary.sort_values(by="champion", ascending=False).reset_index(
        drop=True
    )
    df_summary["position"] = df_summary.index + 1

    cols_csv = [
        "position",
        "team",
        "champion",
        "finalists",
        "semi_finalists",
        "quarter_finalists",
        "round_of_16",
        "round_of_32",
        "group_first_place",
        "group_second_place",
        "group_third_place",
    ]

    df_csv = df_summary[cols_csv].copy()
    df_csv = df_summary[cols_csv].rename(
        columns={
            "finalists": "final",
            "semi_finalists": "semifinals",
            "quarter_finalists": "quarterfinals",
        }
    )
    df_csv["team"] = df_csv["team"].replace(TEAM_MAP_EN_TO_PT)
    df_csv.to_csv("data/summary.csv", index=False)
    df_csv.to_csv("docs/csv/previsoes/summary.csv", index=False)

    df_matches = pd.DataFrame(match_stats)
    df_matches["home_team"] = df_matches["home_team"].replace(TEAM_MAP_EN_TO_PT)
    df_matches["away_team"] = df_matches["away_team"].replace(TEAM_MAP_EN_TO_PT)

    wc_teams = [team for group_teams in groups.values() for team in group_teams]
    export_stan_match_csvs(
        teams_list,
        wc_teams,
        df_matches,
        atk,
        dfn,
        rho,
        et,
        n_sim=n_sim,
        pair_goals_cache=pair_goals_cache,
    )

    # Deterministic display bracket based on the most likely advancement paths.

    group_winners = {}
    group_runners_up = {}
    third_places = []

    for _g_idx, (g_name, g_teams) in enumerate(groups.items()):
        g_df = df_summary[df_summary["team"].isin(g_teams)].copy()

        first = g_df.sort_values(by="group_first_place", ascending=False).iloc[0][
            "team"
        ]
        group_winners[g_name] = first
        g_df = g_df[g_df["team"] != first]

        second = g_df.sort_values(by="group_second_place", ascending=False).iloc[0][
            "team"
        ]
        group_runners_up[g_name] = second
        g_df = g_df[g_df["team"] != second]

        third = g_df.sort_values(by="group_third_place", ascending=False).iloc[0][
            "team"
        ]
        prob_adv = df_summary[df_summary["team"] == third]["round_of_32"].values[0]
        third_places.append({"g_name": g_name, "team": third, "prob": prob_adv})

    best_thirds = sorted(third_places, key=lambda x: x["prob"], reverse=True)[:8]
    best_thirds_groups = [t["g_name"] for t in best_thirds]

    valid_groups_for_1st = {
        "E": ["A", "B", "C", "D", "F"],
        "I": ["C", "D", "F", "G", "H"],
        "D": ["B", "E", "F", "I", "J"],
        "G": ["A", "E", "H", "I", "J"],
        "A": ["C", "E", "F", "H", "I"],
        "L": ["E", "H", "I", "J", "K"],
        "B": ["E", "F", "G", "I", "J"],
        "K": ["D", "E", "I", "J", "L"],
    }

    assigned_thirds = dict.fromkeys(valid_groups_for_1st, "TBD")
    slots = list(valid_groups_for_1st)
    allocation = {}

    def allocate_thirds(index, available_thirds):
        if index == len(slots):
            return True
        slot = slots[index]
        for t_group in available_thirds:
            if t_group in valid_groups_for_1st[slot]:
                allocation[slot] = t_group
                new_available = available_thirds.copy()
                new_available.remove(t_group)
                if allocate_thirds(index + 1, new_available):
                    return True
                del allocation[slot]
        return False

    if allocate_thirds(0, best_thirds_groups):
        for slot, group_letter in allocation.items():
            team_name = next(
                t["team"] for t in best_thirds if t["g_name"] == group_letter
            )
            assigned_thirds[slot] = team_name

    r32_matches = [
        # Left side display slots (L1 to L8).
        (group_winners["E"], assigned_thirds["E"]),
        (group_winners["I"], assigned_thirds["I"]),
        (group_runners_up["A"], group_runners_up["B"]),
        (group_winners["F"], group_runners_up["C"]),
        (group_runners_up["K"], group_runners_up["L"]),
        (group_winners["H"], group_runners_up["J"]),
        (group_winners["D"], assigned_thirds["D"]),
        (group_winners["G"], assigned_thirds["G"]),
        # Right side display slots (R1 to R8).
        (group_winners["C"], group_runners_up["F"]),
        (group_runners_up["E"], group_runners_up["I"]),
        (group_winners["A"], assigned_thirds["A"]),
        (group_winners["L"], assigned_thirds["L"]),
        (group_winners["J"], group_runners_up["H"]),
        (group_runners_up["D"], group_runners_up["G"]),
        (group_winners["B"], assigned_thirds["B"]),
        (group_winners["K"], assigned_thirds["K"]),
    ]

    order_map = {
        "L1": 1,
        "L2": 2,
        "L3": 3,
        "L4": 4,
        "L5": 5,
        "L6": 6,
        "L7": 7,
        "L8": 8,
        "RL1": 9,
        "RL2": 10,
        "RL3": 11,
        "RL4": 12,
        "QL1": 13,
        "QL2": 14,
        "SL": 15,
        "R1": 16,
        "R2": 17,
        "R3": 18,
        "R4": 19,
        "R5": 20,
        "R6": 21,
        "R7": 22,
        "R8": 23,
        "RR1": 24,
        "RR2": 25,
        "RR3": 26,
        "RR4": 27,
        "QR1": 28,
        "QR2": 29,
        "SR": 30,
        "F": 31,
        "T": 32,
    }

    final_rounds = [
        (
            "R32",
            0,
            r32_matches,
            [
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
                "L6",
                "L7",
                "L8",
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
            ],
        ),
        ("Oitavas", 1, [], ["RL1", "RL2", "RL3", "RL4", "RR1", "RR2", "RR3", "RR4"]),
        ("Quartas", 2, [], ["QL1", "QL2", "QR1", "QR2"]),
        ("Semifinal", 3, [], ["SL", "SR"]),
        ("3º Lugar", 4, [], ["T"]),
        ("Final", 4, [], ["F"]),
    ]

    bracket_data = []

    for i, (round_label, round_index, matches, ids) in enumerate(final_rounds):
        next_matches_teams = []
        next_matches_losers = []

        for m_idx, match in enumerate(matches):
            t1, t2 = match
            match_id = ids[m_idx]

            if match_id == "T":
                side = "terceiro"
            elif "L" in match_id:
                side = "left"
            elif "R" in match_id:
                side = "right"
            else:
                side = "final"

            if t1 == "TBD" or t2 == "TBD":
                prob_t1, prob_t2 = (100.0, 0.0) if t1 != "TBD" else (0.0, 100.0)
                winner = t1 if t1 != "TBD" else t2
                loser = t2 if winner == t1 else t1
            else:
                idx1, idx2 = t_to_idx[t1], t_to_idx[t2]
                la, lb = _match_lambdas(atk, dfn, idx1, idx2, et)
                la = la.reshape(-1, 1)
                lb = lb.reshape(-1, 1)
                rho_exp = rho.reshape(-1, 1) if uses_dixon_coles else None

                ga, gb = simulate_matches(la, lb, rho_exp, n_sim)
                # prob_t1 = np.mean(ga > gb) * 100
                # prob_t2 = np.mean(gb > ga) * 100
                prob_t1 = np.mean(ga > gb) * 100
                prob_draw = np.mean(ga == gb) * 100
                prob_t2 = np.mean(gb > ga) * 100

                # Chance de avançar = vitória no tempo normal
                # + (empate no tempo normal × 50% em prorrogação/pênaltis)
                prob_adv_t1 = prob_t1 + (prob_draw * 0.5)
                prob_adv_t2 = 100 - prob_adv_t1

                # winner = t1 if prob_t1 >= prob_t2 else t2
                # loser = t2 if prob_t1 >= prob_t2 else t1
                winner = t1 if prob_adv_t1 >= prob_adv_t2 else t2
                loser = t2 if winner == t1 else t1

            next_matches_teams.append(winner)
            next_matches_losers.append(loser)
            winner_side = "home" if prob_t1 >= prob_t2 else "away"

            bracket_data.append(
                {
                    "side": side,
                    "round_index": round_index,
                    "round_label": round_label,
                    "order": order_map[match_id],
                    "id": match_id,
                    "home_team": t1,
                    "prob_home": prob_adv_t1,
                    "away_team": t2,
                    "prob_away": prob_adv_t2,
                    "winner": winner_side,
                }
            )

        if round_label == "Semifinal":
            final_rounds[i + 1][2].append(
                (next_matches_losers[0], next_matches_losers[1])
            )
            final_rounds[i + 2][2].append(
                (next_matches_teams[0], next_matches_teams[1])
            )
        elif round_label not in ["Semifinal", "3º Lugar", "Final"]:
            final_rounds[i + 1][2].extend(
                list(
                    zip(
                        next_matches_teams[0::2], next_matches_teams[1::2], strict=False
                    )
                )
            )

    bracket_df = (
        pd.DataFrame(bracket_data).sort_values(by="order").reset_index(drop=True)
    )
    bracket_df = bracket_df[
        [
            "side",
            "round_index",
            "round_label",
            "order",
            "id",
            "home_team",
            "prob_home",
            "away_team",
            "prob_away",
            "winner",
        ]
    ]

    bracket_df["prob_home"] = bracket_df["prob_home"].round(2)
    bracket_df["prob_away"] = bracket_df["prob_away"].round(2)
    bracket_df["home_team"] = bracket_df["home_team"].replace(TEAM_MAP_EN_TO_PT)
    bracket_df["away_team"] = bracket_df["away_team"].replace(TEAM_MAP_EN_TO_PT)
    bracket_df.to_csv("docs/csv/previsoes/chaveamento_probs.csv", index=False)
    bracket_df.to_csv("data/chaveamento_probs.csv", index=False)

    probs = {stage: stats[stage] / n_sim for stage in original_stages}
    return probs
