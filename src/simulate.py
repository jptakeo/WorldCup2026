import numpy as np
import pandas as pd
from src.constants import TEAM_MAP_EN_TO_PT, TEAM_MAP_PT_TO_EN

# O Segredo da Velocidade: Calculamos o fatorial na mão para evitar a lentidão das bibliotecas!
# Vai de 0! (1) até 10! (3628800)
FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800])

df_jogos_reais = pd.read_csv('data/world_cup_results.csv')


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


# def simular_copa_2026(post_draws, teams_list, grupos, df_schedule=df_jogos_reais, n_sim=10_000):
#     atk_draws, dfn_draws, eta_draws = post_draws['attack'], post_draws['defense'], post_draws['eta']
#     usa_dixon_coles = 'rho' in post_draws
#     rho_draws = post_draws['rho'] if usa_dixon_coles else None

#     n_samples, n_teams = len(eta_draws), len(teams_list)
    
#     # Dicionário original de Nomes -> IDs
#     t_to_idx = {name: i for i, name in enumerate(teams_list)}
#     g_indices = np.array([[t_to_idx[t] for t in ts] for ts in grupos.values()])

#     idx_samples = np.random.choice(n_samples, n_sim)
#     atk, dfn = atk_draws[idx_samples], dfn_draws[idx_samples]
#     et = eta_draws[idx_samples].reshape(-1, 1)
#     rho = rho_draws[idx_samples] if usa_dixon_coles else None

#     fases_originais = ["avancou_grupos", "round_of_32", "round_of_16", 
#                        "quarter_finalists", "semi_finalists", "finalists", "champion"]
#     todas_fases = fases_originais + ["group_first_place", "group_second_place", "group_third_place"]
    
#     stats = {f: np.zeros(n_teams) for f in todas_fases}

#     # --- ADICIONANDO IDs AO DATAFRAME DE JOGOS ---
#     name_to_idx = {name: idx for name, idx in t_to_idx.items()}
    
#     df_schedule['home_id'] = df_schedule['home_team'].map(TEAM_MAP_PT_TO_EN).map(name_to_idx).astype('Int64')
#     df_schedule['away_id'] = df_schedule['away_team'].map(TEAM_MAP_PT_TO_EN).map(name_to_idx).astype('Int64')

#     # print(df_schedule.head())
    
#     date_map = {}
#     for _, row in df_schedule.iterrows():
#         # print(row['home_id'])
#         if pd.notna(row['home_id']) and pd.notna(row['away_id']):
#             match_key = frozenset([row['home_id'], row['away_id']])
#             date_map[match_key] = row['date']
#     # print(date_map)
#     # --- FASE DE GRUPOS ---
#     match_stats = []
#     num_to_word = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
    
#     pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
#     pts, sg, gp = np.zeros((n_sim, 12, 4)), np.zeros((n_sim, 12, 4)), np.zeros((n_sim, 12, 4))

#     for p1, p2 in pairs:
#         i1, i2 = g_indices[:, p1], g_indices[:, p2]
#         l1 = np.exp(atk[:, i1] - dfn[:, i2] + et)
#         l2 = np.exp(atk[:, i2] - dfn[:, i1] + et)
#         rho_exp = np.repeat(rho[:, None], 12, axis=1) if usa_dixon_coles else None

#         g1, g2 = simular_jogos(l1, l2, rho_exp, n_sim)

#         for g in range(12):
#             t1_idx, t2_idx = g_indices[g, p1], g_indices[g, p2]
#             team1, team2 = teams_list[t1_idx], teams_list[t2_idx]
            
#             match_date = date_map.get(frozenset([t1_idx, t2_idx]), "Data não encontrada")
            
#             g1_g, g2_g = g1[:, g], g2[:, g]
            
#             match_info = {
#                 'group': list(grupos.keys())[g],
#                 'home_team': team1,
#                 'away_team': team2,
#                 'date': match_date,
#                 'home_win': np.mean(g1_g > g2_g) * 100,
#                 'draw': np.mean(g1_g == g2_g) * 100,
#                 'away_win': np.mean(g1_g < g2_g) * 100,
#                 'home_real': None,
#                 'away_real': None
#             }
            
#             for i in range(5):
#                 for j in range(5):
#                     col_name = f"{num_to_word[i]}_{num_to_word[j]}"
#                     match_info[col_name] = np.mean((g1_g == i) & (g2_g == j)) * 100
                    
#             match_stats.append(match_info)

#         pts[:, :, p1] += (g1 > g2)*3 + (g1 == g2)
#         pts[:, :, p2] += (g2 > g1)*3 + (g1 == g2)
#         sg[:, :, p1] += (g1 - g2)
#         sg[:, :, p2] += (g2 - g1)
#         gp[:, :, p1] += g1
#         gp[:, :, p2] += g2

#     sort_val = pts * 1000000 + sg * 1000 + gp
#     ranks = np.argsort(-sort_val, axis=2)

#     classificados = np.zeros((n_sim, 32), dtype=int)
#     terceiros_val = np.zeros((n_sim, 12))

#     def contar_fase(array_fase, nome_fase):
#         unique, counts = np.unique(array_fase, return_counts=True)
#         stats[nome_fase][unique] += counts

#     for g in range(12):
#         classificados[:, g*2] = g_indices[g, ranks[:, g, 0]]
#         classificados[:, g*2+1] = g_indices[g, ranks[:, g, 1]]
#         terceiros_val[:, g] = sort_val[np.arange(n_sim), g, ranks[:, g, 2]]
        
#         contar_fase(g_indices[g, ranks[:, g, 0]], "group_first_place")
#         contar_fase(g_indices[g, ranks[:, g, 1]], "group_second_place")
#         contar_fase(g_indices[g, ranks[:, g, 2]], "group_third_place")

#     best_3rd_idx = np.argsort(-terceiros_val, axis=1)[:, :8]
#     row_idx = np.arange(n_sim)[:, None]
#     classificados[:, 24:] = g_indices[best_3rd_idx, ranks[row_idx, best_3rd_idx, 2]]

#     contar_fase(classificados, "avancou_grupos")

#     # --- MATA-MATA (Simulação Monte Carlo) ---
#     def rodada(competidores):
#         n = competidores.shape[1] // 2
#         a, b = competidores[:, 0::2], competidores[:, 1::2]
#         la = np.exp(atk[np.arange(n_sim)[:, None], a] - dfn[np.arange(n_sim)[:, None], b] + et)
#         lb = np.exp(atk[np.arange(n_sim)[:, None], b] - dfn[np.arange(n_sim)[:, None], a] + et)
#         rho_exp = np.repeat(rho[:, None], n, axis=1) if usa_dixon_coles else None
        
#         ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
#         vence_a = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))
#         return np.where(vence_a, a, b)

#     r32 = classificados
#     contar_fase(r32, "round_of_32")

#     r16 = rodada(r32)
#     contar_fase(r16, "round_of_16")

#     qf = rodada(r16)
#     contar_fase(qf, "quarter_finalists")

#     sf = rodada(qf)
#     contar_fase(sf, "semi_finalists")

#     fin = rodada(sf)
#     contar_fase(fin, "finalists")

#     champ = rodada(fin)
#     contar_fase(champ, "champion")

#     # --- EXPORTAÇÃO PARA CSV (Sumário e Partidas) ---
#     df_summary = pd.DataFrame(stats)
#     df_summary = (df_summary / n_sim * 100).round(2)
#     df_summary.insert(0, 'team', teams_list)
    
#     times_nos_grupos = [time for lista_times in grupos.values() for time in lista_times]
#     df_summary = df_summary[df_summary['team'].isin(times_nos_grupos)]
#     df_summary = df_summary.sort_values(by='champion', ascending=False).reset_index(drop=True)
#     df_summary['position'] = df_summary.index + 1
    
#     cols_csv = ['position', 'team', 'champion', 'finalists', 'semi_finalists', 'quarter_finalists', 
#                 'round_of_16', 'round_of_32', 'group_first_place', 'group_second_place', 'group_third_place']
    
#     df_csv = df_summary[cols_csv].copy()
#     df_csv = df_summary[cols_csv].rename(columns={
#         'finalists': 'final',
#         'semi_finalists': 'semifinals',
#         'quarter_finalists': 'quarterfinals'
#     })
#     df_csv['team'] = df_csv['team'].replace(TEAM_MAP_EN_TO_PT)
#     df_csv.to_csv("data/summary.csv", index=False)

#     df_matches = pd.DataFrame(match_stats).round(4)
#     df_matches['home_team'] = df_matches['home_team'].replace(TEAM_MAP_EN_TO_PT)
#     df_matches['away_team'] = df_matches['away_team'].replace(TEAM_MAP_EN_TO_PT)
#     df_matches.to_csv("data/probs_fase_de_grupos.csv", index=False)
#     df_matches.to_csv("docs/csv/previsoes/partidas.csv", index=False)

#     # ======================================================================
#     # --- CHAVEAMENTO DETERMINÍSTICO (NOVO FORMATO DE COLUNAS/ORDEM) ---
#     # ======================================================================
    
#     group_winners = {}
#     group_runners_up = {}
#     third_places = []
    
    
#     for g_idx, (g_name, g_teams) in enumerate(grupos.items()):
#         g_df = df_summary[df_summary['team'].isin(g_teams)].copy()
        
#         first = g_df.sort_values(by='group_first_place', ascending=False).iloc[0]['team']
#         group_winners[g_name] = first
#         g_df = g_df[g_df['team'] != first]
        
#         second = g_df.sort_values(by='group_second_place', ascending=False).iloc[0]['team']
#         group_runners_up[g_name] = second
#         g_df = g_df[g_df['team'] != second]
        
#         third = g_df.sort_values(by='group_third_place', ascending=False).iloc[0]['team']
#         prob_adv = df_summary[df_summary['team'] == third]['round_of_32'].values[0]
#         third_places.append({g_name: third, 'team': third, 'prob': prob_adv})
        
#     best_thirds = [t['team'] for t in sorted(third_places, key=lambda x: x['prob'], reverse=True)[:8]]
#     print(group_winners)
#     print(group_runners_up)
#     print(third_places)
    
#     # 16 matches for R32 (ordered to match 0-7 = L1-L8, 8-15 = R1-R8)
#     r32_matches = [
#         (group_winners['E'], None),
#         (group_winners['I'], None),
#         (group_runners_up['A'], group_runners_up['B']),
#         (group_winners['F'], group_runners_up['C']),
#         (group_runners_up['K'], group_runners_up['L']),
#         (group_winners['H'], group_runners_up['J']),
#         (group_winners['D'], None),
#         (group_winners['G'], None),
#         (group_winners['C'], group_runners_up['F']),
#         (group_runners_up['E'], group_runners_up['I']),
#         (group_winners['A'], None),
#         (group_winners['L'], None),
#         (group_winners['J'], group_runners_up['H']),
#         (group_runners_up['D'], group_runners_up['G']),
#         (group_winners['B'], None),
#         (group_winners['K'], None)
#     ]
    
#     # Mapeamentos Exatos Solicitados (Final = 31, Terceiro = 32)
#     order_map = {
#         'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'L6': 6, 'L7': 7, 'L8': 8,
#         'RL1': 9, 'RL2': 10, 'RL3': 11, 'RL4': 12, 'QL1': 13, 'QL2': 14, 'SL': 15,
#         'R1': 16, 'R2': 17, 'R3': 18, 'R4': 19, 'R5': 20, 'R6': 21, 'R7': 22, 'R8': 23,
#         'RR1': 24, 'RR2': 25, 'RR3': 26, 'RR4': 27, 'QR1': 28, 'QR2': 29, 'SR': 30, 'F': 31, 'T': 32
#     }
    
#     fase_advancement_target = {
#         'R32': 'round_of_16',
#         'Oitavas': 'quarter_finalists',
#         'Quartas': 'semi_finalists',
#         'Semifinal': 'finalists',
#         '3º Lugar': 'third_place',
#         'Final': 'champion'
#     }

#     # Atualizado '3º Lugar' com round_index = 4
#     fases_finais = [
#         ('R32', 0, r32_matches, ['L1','L2','L3','L4','L5','L6','L7','L8', 'R1','R2','R3','R4','R5','R6','R7','R8']),
#         ('Oitavas', 1, [], ['RL1','RL2','RL3','RL4', 'RR1','RR2','RR3','RR4']),
#         ('Quartas', 2, [], ['QL1','QL2', 'QR1','QR2']),
#         ('Semifinal', 3, [], ['SL', 'SR']),
#         ('3º Lugar', 4, [], ['T']),
#         ('Final', 4, [], ['F'])
#     ]
    
#     chaveamento_dados = []
    
#     for i, (round_label, round_index, matches, ids) in enumerate(fases_finais):
#         next_matches_teams = []
#         next_matches_losers = []
        
#         for m_idx, match in enumerate(matches):
#             t1, t2 = match
#             match_id = ids[m_idx]
            
#             # Determinando o Side (left, right, final, terceiro) baseado no ID
#             if match_id == 'T':
#                 side = 'terceiro'
#             elif 'L' in match_id:
#                 side = 'left'
#             elif 'R' in match_id:
#                 side = 'right'
#             else:
#                 side = 'final'
            
#             # Calculando a probabilidade Head-to-Head real do confronto
#             if t1 == "TBD" or t2 == "TBD":
#                 prob_t1 = 100.0 if t1 != "TBD" else 0.0
#                 prob_t2 = 100.0 if t2 != "TBD" else 0.0
#                 winner = t1 if t1 != "TBD" else t2
#                 loser = t2 if winner == t1 else t1
#             else:
#                 idx1 = t_to_idx[t1]
#                 idx2 = t_to_idx[t2]
                
#                 # Força do ataque/defesa para o confronto direto
#                 la = np.exp(atk[:, idx1] - dfn[:, idx2] + et[:, 0]).reshape(-1, 1)
#                 lb = np.exp(atk[:, idx2] - dfn[:, idx1] + et[:, 0]).reshape(-1, 1)
#                 rho_exp = rho.reshape(-1, 1) if usa_dixon_coles else None
                
#                 # Simula o jogo 10.000 vezes usando as distribuições já sorteadas
#                 ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
                
#                 # Probabilidade Pura de Vitória (sem pênaltis / ignorando empates)
#                 prob_t1 = np.mean(ga > gb) * 100
#                 prob_t2 = np.mean(gb > ga) * 100
                
#                 # Avança a equipe com maior probabilidade de vencer no tempo normal
#                 winner = t1 if prob_t1 >= prob_t2 else t2
#                 loser = t2 if prob_t1 >= prob_t2 else t1

#             next_matches_teams.append(winner)
#             next_matches_losers.append(loser)

#             # Determina a flag 'home' ou 'away' para o CSV
#             winner_side = 'home' if prob_t1 >= prob_t2 else 'away'
            
#             chaveamento_dados.append({
#                 'side': side,
#                 'round_index': round_index,
#                 'round_label': round_label,
#                 'order': order_map[match_id],
#                 'id': match_id,
#                 'home_team': t1,
#                 'prob_home': prob_t1,
#                 'away_team': t2,
#                 'prob_away': prob_t2,
#                 'winner': winner_side
#             })
            
#         # Lógica de chaveamento direcionado para contemplar o 3º lugar e a Final
#         if round_label == 'Semifinal':
#             # Manda os perdedores para o 3º Lugar e os vencedores para a Final
#             fases_finais[i+1][2].append((next_matches_losers[0], next_matches_losers[1]))
#             fases_finais[i+2][2].append((next_matches_teams[0], next_matches_teams[1]))
#         elif round_label not in ['Semifinal', '3º Lugar', 'Final']:
#             # Lógica normal para as outras fases
#             fases_finais[i+1][2].extend(list(zip(next_matches_teams[0::2], next_matches_teams[1::2])))
            
#     # Cria o dataframe, ordena estritamente por "order" e garante a sequência das colunas
#     df_chaveamento = pd.DataFrame(chaveamento_dados)
#     df_chaveamento = df_chaveamento.sort_values(by='order').reset_index(drop=True)
    
#     cols_requeridas = ['side', 'round_index', 'round_label', 'order', 'id', 'home_team', 'prob_home', 'away_team', 'prob_away', 'winner']
#     df_chaveamento = df_chaveamento[cols_requeridas]
    
#     df_chaveamento['prob_home'] = df_chaveamento['prob_home'].round(2)
#     df_chaveamento['prob_away'] = df_chaveamento['prob_away'].round(2)
#     df_chaveamento['home_team'] = df_chaveamento['home_team'].replace(TEAM_MAP_EN_TO_PT)
#     df_chaveamento['away_team'] = df_chaveamento['away_team'].replace(TEAM_MAP_EN_TO_PT)
#     df_chaveamento.to_csv("docs/csv/previsoes/chaveamento_probs.csv", index=False)
#     df_chaveamento.to_csv("data/chaveamento_probs.csv", index=False)

#     probs = {fase: stats[fase] / n_sim for fase in fases_originais}
#     return probs


def simular_fase_e_restante(post_draws, teams_list, df_partidas, fases_nomes=None, n_sim=10_000):
    """
    Simula uma fase específica e todas as subsequentes até o campeão.
    O df_partidas deve estar ordenado em formato de chaveamento (0 enfrenta 1, 2 enfrenta 3...).
    """
    atk_draws, dfn_draws, eta_draws = post_draws['attack'], post_draws['defense'], post_draws['eta']
    usa_dixon_coles = 'rho' in post_draws
    rho_draws = post_draws['rho'] if usa_dixon_coles else None

    n_samples, n_teams_total = len(eta_draws), len(teams_list)
    t_to_idx = {name: i for i, name in enumerate(teams_list)}

    idx_samples = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[idx_samples], dfn_draws[idx_samples]
    et = eta_draws[idx_samples].reshape(-1, 1)
    rho = rho_draws[idx_samples] if usa_dixon_coles else None

    num_matches = len(df_partidas)
    
    # Auto-detecta os nomes das fases com base no número de partidas iniciais
    if fases_nomes is None:
        mapeamento_fases = {
            16: ["round_of_32", "round_of_16", "quarter_finalists", "semi_finalists", "finalists", "champion"],
            8: ["round_of_16", "quarter_finalists", "semi_finalists", "finalists", "champion"],
            4: ["quarter_finalists", "semi_finalists", "finalists", "champion"],
            2: ["semi_finalists", "finalists", "champion"],
            1: ["finalists", "champion"]
        }
        # Fallback genérico caso seja um formato não padrão
    fases_nomes = mapeamento_fases.get(num_matches, ["fase_atual"] + [f"avanco_{i}" for i in range(1, int(np.log2(num_matches))+2)])

    stats = {f: np.zeros(n_teams_total) for f in fases_nomes}
    match_stats = []
    num_to_word = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}

    # Array que controlará quem está vivo na competição, shape: (10000, n_times_nesta_fase)
    competidores = np.zeros((n_sim, num_matches * 2), dtype=int)
    
    times_envolvidos = []
    for m in range(num_matches):
        row = df_partidas.iloc[m]
        t1, t2 = row['home_team'], row['away_team']
        times_envolvidos.extend([t1, t2])
        i1, i2 = t_to_idx[t1], t_to_idx[t2]
        
        competidores[:, m*2] = i1
        competidores[:, m*2+1] = i2
        
        # Registra 100% de presença (n_sim) na fase atual inicial para esses times
        stats[fases_nomes[0]][i1] = n_sim
        stats[fases_nomes[0]][i2] = n_sim

    fase_idx = 1
    is_first_round = True
    
    # Loop vetorial que simula as rodadas em cascata até sobrar apenas 1 time
    while competidores.shape[1] > 1:
        n = competidores.shape[1] // 2
        a, b = competidores[:, 0::2], competidores[:, 1::2]
        
        la = np.exp(atk[np.arange(n_sim)[:, None], a] - dfn[np.arange(n_sim)[:, None], b] + et)
        lb = np.exp(atk[np.arange(n_sim)[:, None], b] - dfn[np.arange(n_sim)[:, None], a] + et)
        rho_exp = np.repeat(rho[:, None], n, axis=1) if usa_dixon_coles else None
        
        # Simula todos os jogos da rodada atual de uma vez (presume-se que simular_jogos() exista no ambiente)
        ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
        
        vence_a = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))
        
        # Na primeira rodada (a do DataFrame), extraímos as estatísticas completas jogo a jogo
        if is_first_round:
            for m in range(num_matches):
                g1, g2 = ga[:, m], gb[:, m]
                
                match_info = {
                    'home_team': df_partidas.iloc[m]['home_team'],
                    'away_team': df_partidas.iloc[m]['away_team'],
                    'date': df_partidas.iloc[m].get('date', "Data não encontrada"),
                    'home_win': np.mean(g1 > g2) * 100,
                    'draw': np.mean(g1 == g2) * 100,
                    'away_win': np.mean(g1 < g2) * 100
                }
                
                for i in range(5):
                    for j in range(5):
                        col_name = f"{num_to_word[i]}_{num_to_word[j]}"
                        match_info[col_name] = np.mean((g1 == i) & (g2 == j)) * 100
                        
                match_stats.append(match_info)
            is_first_round = False
        
        # Atualiza os competidores com a metade que venceu
        competidores = np.where(vence_a, a, b)
        
        # Contabiliza a entrada na nova fase
        if fase_idx < len(fases_nomes):
            unique, counts = np.unique(competidores, return_counts=True)
            stats[fases_nomes[fase_idx]][unique] += counts
            
        fase_idx += 1

    # --- MONTAGEM DOS RETORNOS ---
    probs = {fase: stats[fase] / n_sim for fase in fases_nomes}
    
    df_summary = pd.DataFrame(stats)
    df_summary = (df_summary / n_sim * 100).round(2)
    df_summary.insert(0, 'team', teams_list)
    
    # Mantém no sumário final apenas as equipes que existiam no DataFrame inicial
    df_summary = df_summary[df_summary['team'].isin(times_envolvidos)]
    
    # Ordena com base na probabilidade de ser campeão (ou última fase calculada)
    sort_col = fases_nomes[-1]
    df_summary = df_summary.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    df_summary.insert(0, 'position', df_summary.index + 1)

    df_csv = pd.DataFrame({
        'position': df_summary['position'],
        'team': df_summary['team'],
        'champion': df_summary.get('champion', pd.Series([100.0]*len(df_summary))),
        'final': df_summary.get('finalists', pd.Series([100.0]*len(df_summary))),
        'semifinals': df_summary.get('semi_finalists', pd.Series([100.0]*len(df_summary))),
        'quarterfinals': df_summary.get('quarter_finalists', pd.Series([100.0]*len(df_summary))),
        'round_of_16': df_summary.get('round_of_16', pd.Series([100.0]*len(df_summary))),
        'round_of_32': df_summary.get('round_of_32', pd.Series([100.0]*len(df_summary)))
    })
    df_csv['team'] = df_csv['team'].replace(TEAM_MAP_EN_TO_PT)

    df_matches = pd.DataFrame(match_stats).round(2)
    df_matches.reset_index(drop=True, inplace=True)
    df_matches['home_team'] = df_matches['home_team'].replace(TEAM_MAP_EN_TO_PT)
    df_matches['away_team'] = df_matches['away_team'].replace(TEAM_MAP_EN_TO_PT)

    return probs, df_csv, df_matches



def simular_copa_2026(
    post_draws, 
    teams_list, 
    grupos, 
    df_schedule = df_jogos_reais, 
    TEAM_MAP_PT_TO_EN = TEAM_MAP_PT_TO_EN, 
    TEAM_MAP_EN_TO_PT = TEAM_MAP_EN_TO_PT, 
    n_sim=10_000):
    atk_draws, dfn_draws, eta_draws = post_draws['attack'], post_draws['defense'], post_draws['eta']
    usa_dixon_coles = 'rho' in post_draws
    rho_draws = post_draws['rho'] if usa_dixon_coles else None

    n_samples, n_teams = len(eta_draws), len(teams_list)
    
    # Dicionário original de Nomes -> IDs
    t_to_idx = {name: i for i, name in enumerate(teams_list)}
    g_indices = np.array([[t_to_idx[t] for t in ts] for ts in grupos.values()])

    idx_samples = np.random.choice(n_samples, n_sim)
    atk, dfn = atk_draws[idx_samples], dfn_draws[idx_samples]
    et = eta_draws[idx_samples].reshape(-1, 1)
    rho = rho_draws[idx_samples] if usa_dixon_coles else None

    fases_originais = ["avancou_grupos", "round_of_32", "round_of_16", 
                       "quarter_finalists", "semi_finalists", "finalists", "champion"]
    todas_fases = fases_originais + ["group_first_place", "group_second_place", "group_third_place"]
    
    stats = {f: np.zeros(n_teams) for f in todas_fases}

    # --- ADICIONANDO IDs AO DATAFRAME DE JOGOS ---
    name_to_idx = {name: idx for name, idx in t_to_idx.items()}
    
    df_schedule['home_id'] = df_schedule['home_team'].map(TEAM_MAP_PT_TO_EN).map(name_to_idx).astype('Int64')
    df_schedule['away_id'] = df_schedule['away_team'].map(TEAM_MAP_PT_TO_EN).map(name_to_idx).astype('Int64')

    date_map = {}
    for _, row in df_schedule.iterrows():
        if pd.notna(row['home_id']) and pd.notna(row['away_id']):
            match_key = frozenset([row['home_id'], row['away_id']])
            date_map[match_key] = row['date']

    # --- FASE DE GRUPOS ---
    match_stats = []
    num_to_word = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
    
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pts, sg, gp = np.zeros((n_sim, 12, 4)), np.zeros((n_sim, 12, 4)), np.zeros((n_sim, 12, 4))

    for p1, p2 in pairs:
        i1, i2 = g_indices[:, p1], g_indices[:, p2]
        l1 = np.exp(atk[:, i1] - dfn[:, i2] + et)
        l2 = np.exp(atk[:, i2] - dfn[:, i1] + et)
        rho_exp = np.repeat(rho[:, None], 12, axis=1) if usa_dixon_coles else None

        g1, g2 = simular_jogos(l1, l2, rho_exp, n_sim)

        for g in range(12):
            t1_idx, t2_idx = g_indices[g, p1], g_indices[g, p2]
            team1, team2 = teams_list[t1_idx], teams_list[t2_idx]
            match_date = date_map.get(frozenset([t1_idx, t2_idx]), "Data não encontrada")
            g1_g, g2_g = g1[:, g], g2[:, g]
            
            match_info = {
                'group': list(grupos.keys())[g],
                'home_team': team1, 'away_team': team2, 'date': match_date,
                'home_win': np.mean(g1_g > g2_g) * 100, 'draw': np.mean(g1_g == g2_g) * 100, 'away_win': np.mean(g1_g < g2_g) * 100
            }
            for i in range(5):
                for j in range(5):
                    match_info[f"{num_to_word[i]}_{num_to_word[j]}"] = np.mean((g1_g == i) & (g2_g == j)) * 100
            match_stats.append(match_info)

        pts[:, :, p1] += (g1 > g2)*3 + (g1 == g2)
        pts[:, :, p2] += (g2 > g1)*3 + (g1 == g2)
        sg[:, :, p1] += (g1 - g2)
        sg[:, :, p2] += (g2 - g1)
        gp[:, :, p1] += g1
        gp[:, :, p2] += g2

    sort_val = pts * 1000000 + sg * 1000 + gp
    ranks = np.argsort(-sort_val, axis=2)

    def contar_fase(array_fase, nome_fase):
        unique, counts = np.unique(array_fase, return_counts=True)
        stats[nome_fase][unique] += counts

    # Extraindo as classificações de forma vetorizada
    firsts = np.zeros((n_sim, 12), dtype=int)
    seconds = np.zeros((n_sim, 12), dtype=int)
    all_thirds = np.zeros((n_sim, 12), dtype=int)
    terceiros_val = np.zeros((n_sim, 12))

    for g in range(12):
        firsts[:, g] = g_indices[g, ranks[:, g, 0]]
        seconds[:, g] = g_indices[g, ranks[:, g, 1]]
        all_thirds[:, g] = g_indices[g, ranks[:, g, 2]]
        terceiros_val[:, g] = sort_val[np.arange(n_sim), g, ranks[:, g, 2]]
        
        contar_fase(firsts[:, g], "group_first_place")
        contar_fase(seconds[:, g], "group_second_place")
        contar_fase(all_thirds[:, g], "group_third_place")

    # Mapeando os 8 melhores terceiros
    best_3rd_group_idx = np.argsort(-terceiros_val, axis=1)[:, :8]
    sorted_thirds = np.sort(best_3rd_group_idx, axis=1)

    # Regras e Slots para simulação (Mapemanto de Grupos: 0=A, 1=B, ..., 11=L)
    valid_groups_for_1st_idx = {
        4: [0, 1, 2, 3, 5],   # 1E
        8: [2, 3, 5, 6, 7],   # 1I
        3: [1, 4, 5, 8, 9],   # 1D
        6: [0, 4, 7, 8, 9],   # 1G
        0: [2, 4, 5, 7, 8],   # 1A
        11: [4, 7, 8, 9, 10], # 1L
        1: [4, 5, 6, 8, 9],   # 1B
        10: [3, 4, 8, 9, 11]  # 1K
    }
    slots_idx = [4, 8, 3, 6, 0, 11, 1, 10]
    allocation_cache = {}

    def get_allocation(thirds_tuple):
        if thirds_tuple in allocation_cache: return allocation_cache[thirds_tuple]
        thirds_list = list(thirds_tuple)
        alloc = {}
        
        def backtrack(idx, available):
            if idx == 8: return True
            slot = slots_idx[idx]
            for t in available:
                if t in valid_groups_for_1st_idx[slot]:
                    alloc[slot] = t
                    new_avail = available.copy()
                    new_avail.remove(t)
                    if backtrack(idx + 1, new_avail): return True
                    del alloc[slot]
            return False
            
        if backtrack(0, thirds_list):
            res = [alloc[s] for s in slots_idx]
        else:
            res = thirds_list.copy() # Fallback de segurança 
            
        allocation_cache[thirds_tuple] = res
        return res

    allocated_thirds = np.zeros((n_sim, 8), dtype=int)
    for i in range(n_sim):
        allocated_thirds[i] = get_allocation(tuple(sorted_thirds[i]))

    # Construção rigorosa e determinística da Chave (r32)
    row_idx = np.arange(n_sim)
    r32 = np.zeros((n_sim, 32), dtype=int)

    # --- Lado Esquerdo ---
    r32[:, 0], r32[:, 1] = firsts[:, 4], all_thirds[row_idx, allocated_thirds[:, 0]]    # 1E vs 3...
    r32[:, 2], r32[:, 3] = firsts[:, 8], all_thirds[row_idx, allocated_thirds[:, 1]]    # 1I vs 3...
    r32[:, 4], r32[:, 5] = seconds[:, 0], seconds[:, 1]                                 # 2A vs 2B
    r32[:, 6], r32[:, 7] = firsts[:, 5], seconds[:, 2]                                  # 1F vs 2C
    r32[:, 8], r32[:, 9] = seconds[:, 10], seconds[:, 11]                               # 2K vs 2L
    r32[:, 10], r32[:, 11] = firsts[:, 7], seconds[:, 9]                                # 1H vs 2J
    r32[:, 12], r32[:, 13] = firsts[:, 3], all_thirds[row_idx, allocated_thirds[:, 2]]  # 1D vs 3...
    r32[:, 14], r32[:, 15] = firsts[:, 6], all_thirds[row_idx, allocated_thirds[:, 3]]  # 1G vs 3...

    # --- Lado Direito ---
    r32[:, 16], r32[:, 17] = firsts[:, 2], seconds[:, 5]                                # 1C vs 2F
    r32[:, 18], r32[:, 19] = seconds[:, 4], seconds[:, 8]                               # 2E vs 2I
    r32[:, 20], r32[:, 21] = firsts[:, 0], all_thirds[row_idx, allocated_thirds[:, 4]]  # 1A vs 3...
    r32[:, 22], r32[:, 23] = firsts[:, 11], all_thirds[row_idx, allocated_thirds[:, 5]] # 1L vs 3...
    r32[:, 24], r32[:, 25] = firsts[:, 9], seconds[:, 7]                                # 1J vs 2H
    r32[:, 26], r32[:, 27] = seconds[:, 3], seconds[:, 6]                               # 2D vs 2G
    r32[:, 28], r32[:, 29] = firsts[:, 1], all_thirds[row_idx, allocated_thirds[:, 6]]  # 1B vs 3...
    r32[:, 30], r32[:, 31] = firsts[:, 10], all_thirds[row_idx, allocated_thirds[:, 7]] # 1K vs 3...

    contar_fase(r32, "avancou_grupos")
    contar_fase(r32, "round_of_32")

    # --- MATA-MATA (Simulação Monte Carlo) ---
    def rodada(competidores):
        n = competidores.shape[1] // 2
        a, b = competidores[:, 0::2], competidores[:, 1::2]
        la = np.exp(atk[np.arange(n_sim)[:, None], a] - dfn[np.arange(n_sim)[:, None], b] + et)
        lb = np.exp(atk[np.arange(n_sim)[:, None], b] - dfn[np.arange(n_sim)[:, None], a] + et)
        rho_exp = np.repeat(rho[:, None], n, axis=1) if usa_dixon_coles else None
        
        ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
        vence_a = (ga > gb) | ((ga == gb) & (np.random.rand(n_sim, n) < 0.5))
        return np.where(vence_a, a, b)

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

    # --- EXPORTAÇÃO PARA CSV (Sumário e Partidas) ---
    df_summary = pd.DataFrame(stats)
    df_summary = (df_summary / n_sim * 100).round(2)
    df_summary.insert(0, 'team', teams_list)
    
    times_nos_grupos = [time for lista_times in grupos.values() for time in lista_times]
    df_summary = df_summary[df_summary['team'].isin(times_nos_grupos)]
    df_summary = df_summary.sort_values(by='champion', ascending=False).reset_index(drop=True)
    df_summary['position'] = df_summary.index + 1
    
    cols_csv = ['position', 'team', 'champion', 'finalists', 'semi_finalists', 'quarter_finalists', 
                'round_of_16', 'round_of_32', 'group_first_place', 'group_second_place', 'group_third_place']
    
    df_csv = df_summary[cols_csv].copy()
    df_csv = df_summary[cols_csv].rename(columns={
        'finalists': 'final',
        'semi_finalists': 'semifinals',
        'quarter_finalists': 'quarterfinals'
    })
    df_csv['team'] = df_csv['team'].replace(TEAM_MAP_EN_TO_PT)
    df_csv.to_csv("data/summary.csv", index=False)

    df_matches = pd.DataFrame(match_stats).round(4)
    df_matches['home_team'] = df_matches['home_team'].replace(TEAM_MAP_EN_TO_PT)
    df_matches['away_team'] = df_matches['away_team'].replace(TEAM_MAP_EN_TO_PT)
    df_matches.to_csv("data/probs_fase_de_grupos.csv", index=False)
    df_matches.to_csv("docs/csv/previsoes/partidas.csv", index=False)

    # ======================================================================
    # --- CHAVEAMENTO DETERMINÍSTICO (NOVO FORMATO DE COLUNAS/ORDEM) ---
    # ======================================================================
    
    group_winners = {}
    group_runners_up = {}
    third_places = []
    
    for g_idx, (g_name, g_teams) in enumerate(grupos.items()):
        g_df = df_summary[df_summary['team'].isin(g_teams)].copy()
        
        first = g_df.sort_values(by='group_first_place', ascending=False).iloc[0]['team']
        group_winners[g_name] = first
        g_df = g_df[g_df['team'] != first]
        
        second = g_df.sort_values(by='group_second_place', ascending=False).iloc[0]['team']
        group_runners_up[g_name] = second
        g_df = g_df[g_df['team'] != second]
        
        third = g_df.sort_values(by='group_third_place', ascending=False).iloc[0]['team']
        prob_adv = df_summary[df_summary['team'] == third]['round_of_32'].values[0]
        third_places.append({'g_name': g_name, 'team': third, 'prob': prob_adv})
        
    best_thirds = sorted(third_places, key=lambda x: x['prob'], reverse=True)[:8]
    best_thirds_groups = [t['g_name'] for t in best_thirds]
    
    valid_groups_for_1st = {
        'E': ['A', 'B', 'C', 'D', 'F'], 'I': ['C', 'D', 'F', 'G', 'H'],
        'D': ['B', 'E', 'F', 'I', 'J'], 'G': ['A', 'E', 'H', 'I', 'J'],
        'A': ['C', 'E', 'F', 'H', 'I'], 'L': ['E', 'H', 'I', 'J', 'K'],
        'B': ['E', 'F', 'G', 'I', 'J'], 'K': ['D', 'E', 'I', 'J', 'L']
    }
    
    assigned_thirds = {k: "TBD" for k in valid_groups_for_1st.keys()}
    slots = list(valid_groups_for_1st.keys())
    allocation = {}
    
    def allocate_thirds(index, available_thirds):
        if index == len(slots): return True
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
            team_name = next(t['team'] for t in best_thirds if t['g_name'] == group_letter)
            assigned_thirds[slot] = team_name
            
    r32_matches = [
        # Lado esquerdo (L1 a L8)
        (group_winners['E'], assigned_thirds['E']),
        (group_winners['I'], assigned_thirds['I']),
        (group_runners_up['A'], group_runners_up['B']),
        (group_winners['F'], group_runners_up['C']),
        (group_runners_up['K'], group_runners_up['L']),
        (group_winners['H'], group_runners_up['J']),
        (group_winners['D'], assigned_thirds['D']),
        (group_winners['G'], assigned_thirds['G']),

        # Lado direito (R1 a R8)
        (group_winners['C'], group_runners_up['F']),
        (group_runners_up['E'], group_runners_up['I']),
        (group_winners['A'], assigned_thirds['A']),
        (group_winners['L'], assigned_thirds['L']),
        (group_winners['J'], group_runners_up['H']),
        (group_runners_up['D'], group_runners_up['G']),
        (group_winners['B'], assigned_thirds['B']),
        (group_winners['K'], assigned_thirds['K'])
    ]
    
    order_map = {
        'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'L6': 6, 'L7': 7, 'L8': 8,
        'RL1': 9, 'RL2': 10, 'RL3': 11, 'RL4': 12, 'QL1': 13, 'QL2': 14, 'SL': 15,
        'R1': 16, 'R2': 17, 'R3': 18, 'R4': 19, 'R5': 20, 'R6': 21, 'R7': 22, 'R8': 23,
        'RR1': 24, 'RR2': 25, 'RR3': 26, 'RR4': 27, 'QR1': 28, 'QR2': 29, 'SR': 30, 'F': 31, 'T': 32
    }

    fases_finais = [
        ('R32', 0, r32_matches, ['L1','L2','L3','L4','L5','L6','L7','L8', 'R1','R2','R3','R4','R5','R6','R7','R8']),
        ('Oitavas', 1, [], ['RL1','RL2','RL3','RL4', 'RR1','RR2','RR3','RR4']),
        ('Quartas', 2, [], ['QL1','QL2', 'QR1','QR2']),
        ('Semifinal', 3, [], ['SL', 'SR']),
        ('3º Lugar', 4, [], ['T']),
        ('Final', 4, [], ['F'])
    ]
    
    chaveamento_dados = []
    
    for i, (round_label, round_index, matches, ids) in enumerate(fases_finais):
        next_matches_teams = []
        next_matches_losers = []
        
        for m_idx, match in enumerate(matches):
            t1, t2 = match
            match_id = ids[m_idx]
            
            if match_id == 'T': side = 'terceiro'
            elif 'L' in match_id: side = 'left'
            elif 'R' in match_id: side = 'right'
            else: side = 'final'
            
            if t1 == "TBD" or t2 == "TBD":
                prob_t1, prob_t2 = (100.0, 0.0) if t1 != "TBD" else (0.0, 100.0)
                winner = t1 if t1 != "TBD" else t2
                loser = t2 if winner == t1 else t1
            else:
                idx1, idx2 = t_to_idx[t1], t_to_idx[t2]
                la = np.exp(atk[:, idx1] - dfn[:, idx2] + et[:, 0]).reshape(-1, 1)
                lb = np.exp(atk[:, idx2] - dfn[:, idx1] + et[:, 0]).reshape(-1, 1)
                rho_exp = rho.reshape(-1, 1) if usa_dixon_coles else None
                
                ga, gb = simular_jogos(la, lb, rho_exp, n_sim)
                prob_t1 = np.mean(ga > gb) * 100
                prob_t2 = np.mean(gb > ga) * 100
                
                winner = t1 if prob_t1 >= prob_t2 else t2
                loser = t2 if prob_t1 >= prob_t2 else t1

            next_matches_teams.append(winner)
            next_matches_losers.append(loser)
            winner_side = 'home' if prob_t1 >= prob_t2 else 'away'
            
            chaveamento_dados.append({
                'side': side, 'round_index': round_index, 'round_label': round_label,
                'order': order_map[match_id], 'id': match_id,
                'home_team': t1, 'prob_home': prob_t1,
                'away_team': t2, 'prob_away': prob_t2,
                'winner': winner_side
            })
            
        if round_label == 'Semifinal':
            fases_finais[i+1][2].append((next_matches_losers[0], next_matches_losers[1]))
            fases_finais[i+2][2].append((next_matches_teams[0], next_matches_teams[1]))
        elif round_label not in ['Semifinal', '3º Lugar', 'Final']:
            fases_finais[i+1][2].extend(list(zip(next_matches_teams[0::2], next_matches_teams[1::2])))
            
    df_chaveamento = pd.DataFrame(chaveamento_dados).sort_values(by='order').reset_index(drop=True)
    df_chaveamento = df_chaveamento[['side', 'round_index', 'round_label', 'order', 'id', 'home_team', 'prob_home', 'away_team', 'prob_away', 'winner']]
    
    df_chaveamento['prob_home'] = df_chaveamento['prob_home'].round(2)
    df_chaveamento['prob_away'] = df_chaveamento['prob_away'].round(2)
    df_chaveamento['home_team'] = df_chaveamento['home_team'].replace(TEAM_MAP_EN_TO_PT)
    df_chaveamento['away_team'] = df_chaveamento['away_team'].replace(TEAM_MAP_EN_TO_PT)
    df_chaveamento.to_csv("docs/csv/previsoes/chaveamento_probs.csv", index=False)
    df_chaveamento.to_csv("data/chaveamento_probs.csv", index=False)

    probs = {fase: stats[fase] / n_sim for fase in fases_originais}
    return probs