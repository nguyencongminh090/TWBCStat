"""Phase 2 — Elo Ratings & Bayesian Head-to-Head"""
import sqlite3, os
import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'twbc.db')
OUT = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

def compute_elo(pairings_df, career_games, max_round=None):
    """Compute Elo from ordered pairings. Returns elo dict and progression."""
    elo = {}
    progression = {}  # nick -> {round: elo}
    K_DEFAULT, K_EXP = 32, 16

    for _, row in pairings_df.iterrows():
        if max_round and row['tournament_round'] > max_round:
            break
        na, nb = row['nick_a'], row['nick_b']
        elo.setdefault(na, 1200.0); elo.setdefault(nb, 1200.0)

        ra, rb = elo[na], elo[nb]
        ea = 1 / (1 + 10**((rb - ra) / 400))

        if row['winner'] == 'A':     sa, sb = 1.0, 0.0
        elif row['winner'] == 'B':   sa, sb = 0.0, 1.0
        else:                        sa, sb = 0.5, 0.5

        ka = K_EXP if career_games.get(na, 0) > 60 else K_DEFAULT
        kb = K_EXP if career_games.get(nb, 0) > 60 else K_DEFAULT
        elo[na] += ka * (sa - ea)
        elo[nb] += kb * (sb - (1 - ea))

        tr = int(row['tournament_round'])
        progression.setdefault(na, {})[tr] = elo[na]
        progression.setdefault(nb, {})[tr] = elo[nb]

    return elo, progression

def run():
    conn = sqlite3.connect(DB)

    # Load ordered pairings
    pairings = pd.read_sql("""
        SELECT vp.pairing_id, m.tournament_round, sr.round_number,
            pa.nick AS nick_a, pb.nick AS nick_b,
            vp.score_a, vp.score_b, vp.total_games, vp.winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id=vp.sub_round_id
        JOIN matches m ON m.match_id=sr.match_id
        JOIN players pa ON pa.player_id=vp.player_a_id
        JOIN players pb ON pb.player_id=vp.player_b_id
        ORDER BY m.tournament_round, sr.round_number, vp.pairing_id
    """, conn)

    career = pd.read_sql("SELECT player_nick, player_name, team, total_games FROM v_player_overall", conn)
    career_games = dict(zip(career['player_nick'], career['total_games']))

    # ── 2.1 Full Elo ──
    print("Step 2.1 — Elo ratings")
    elo_final, prog = compute_elo(pairings, career_games)

    elo_df = pd.DataFrame([
        {'player_nick': k, 'elo': round(v, 1),
         'player_name': career.set_index('player_nick').loc[k, 'player_name'] if k in career['player_nick'].values else k,
         'team': career.set_index('player_nick').loc[k, 'team'] if k in career['player_nick'].values else '?',
         'career_games': career_games.get(k, 0)}
        for k, v in elo_final.items()
    ]).sort_values('elo', ascending=False)
    elo_df.to_csv(f"{OUT}/elo_ratings_final.csv", index=False)
    print("  Top 15 by Elo:")
    print(elo_df.head(15).to_string(index=False))

    # Progression
    rounds = sorted(set(r for p in prog.values() for r in p.keys()))
    prog_rows = []
    for nick, rd in prog.items():
        row = {'player_nick': nick}
        for r in rounds:
            row[f'round_{r}_elo'] = round(rd.get(r, 1200), 1)
        prog_rows.append(row)
    prog_df = pd.DataFrame(prog_rows)
    prog_df.to_csv(f"{OUT}/elo_progression.csv", index=False)

    # ── 2.2 Win probability matrix ──
    print("\nStep 2.2 — Win probability matrix")
    h2h_pairs = pd.read_sql("""
        SELECT DISTINCT pa.nick AS nick_a, pb.nick AS nick_b
        FROM v_pairings vp
        JOIN players pa ON pa.player_id=vp.player_a_id
        JOIN players pb ON pb.player_id=vp.player_b_id
    """, conn)
    all_nicks = sorted(set(h2h_pairs['nick_a']) | set(h2h_pairs['nick_b']))
    wp_matrix = pd.DataFrame(0.5, index=all_nicks, columns=all_nicks)
    for a in all_nicks:
        for b in all_nicks:
            if a != b and a in elo_final and b in elo_final:
                wp_matrix.loc[a, b] = round(1 / (1 + 10**((elo_final[b] - elo_final[a]) / 400)), 4)
    wp_matrix.to_csv(f"{OUT}/win_prob_matrix_elo.csv")

    # ── 2.3 Bayesian head-to-head ──
    print("Step 2.3 — Bayesian head-to-head")
    h2h = pd.read_sql("""
        SELECT pa.nick AS nick_a, pb.nick AS nick_b, COUNT(*) AS games,
            SUM(CASE WHEN vp.winner='A' THEN 1 ELSE 0 END) AS a_wins,
            SUM(CASE WHEN vp.winner='B' THEN 1 ELSE 0 END) AS b_wins,
            SUM(CASE WHEN vp.winner='Draw' THEN 1 ELSE 0 END) AS draws
        FROM v_pairings vp
        JOIN players pa ON pa.player_id=vp.player_a_id
        JOIN players pb ON pb.player_id=vp.player_b_id
        GROUP BY vp.player_a_id, vp.player_b_id HAVING COUNT(*)>=1
    """, conn)
    AP, BP = 2, 2
    h2h['p_a_wins'] = (AP + h2h['a_wins']) / (AP + BP + h2h['a_wins'] + h2h['b_wins'])
    h2h['ci_lower'] = h2h.apply(lambda r: beta_dist.ppf(0.025, AP+r.a_wins, BP+r.b_wins), axis=1)
    h2h['ci_upper'] = h2h.apply(lambda r: beta_dist.ppf(0.975, AP+r.a_wins, BP+r.b_wins), axis=1)
    h2h['uncertainty'] = h2h['ci_upper'] - h2h['ci_lower']
    h2h.to_csv(f"{OUT}/bayesian_head_to_head.csv", index=False)
    print(f"  {len(h2h)} matchups saved")

    # ── 2.4 Visualizations ──
    print("\nStep 2.4 — Visualizations")

    # Heatmap: win prob for players with >= 12 career games
    active = [n for n in all_nicks if career_games.get(n, 0) >= 12]
    if len(active) > 2:
        sub = wp_matrix.loc[active, active]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(sub, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax,
                    linewidths=0.3, annot_kws={'size': 6})
        ax.set_title('Elo Win Probability Matrix (≥12 career games)')
        plt.tight_layout(); fig.savefig(f"{OUT}/heatmap_winprob_elo.png", dpi=DPI); plt.close()

    # Elo progression
    top10 = elo_df.head(10)['player_nick'].tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for nick in top10:
        if nick in prog:
            rd = prog[nick]
            xs = sorted(rd.keys())
            ax.plot(xs, [rd[x] for x in xs], marker='o', label=nick, linewidth=2)
    ax.set_xlabel('Tournament Round'); ax.set_ylabel('Elo Rating')
    ax.set_title('Elo Progression — Top 10 Players'); ax.legend(fontsize=8, ncol=2)
    ax.set_xticks(rounds)
    plt.tight_layout(); fig.savefig(f"{OUT}/plot_elo_progression.png", dpi=DPI); plt.close()

    # Elo vs efficiency
    merged = elo_df.merge(pd.read_csv(f"{OUT}/player_career_stats.csv")[['player_nick','career_efficiency']], on='player_nick')
    teams_unique = merged['team'].unique()
    colors = dict(zip(teams_unique, sns.color_palette('husl', len(teams_unique))))
    fig, ax = plt.subplots(figsize=(10, 6))
    for t in teams_unique:
        sub = merged[merged['team'] == t]
        ax.scatter(sub['career_efficiency'], sub['elo'], s=sub['career_games']*1.5,
                   label=t, color=colors[t], alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Career Efficiency'); ax.set_ylabel('Final Elo')
    ax.set_title('Elo vs Efficiency (size = career games)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    plt.tight_layout(); fig.savefig(f"{OUT}/scatter_elo_vs_efficiency.png", dpi=DPI); plt.close()
    print("  ✓ All Phase 2 plots saved")

    conn.close()
    return elo_final, prog, career_games, pairings

if __name__ == "__main__":
    run()
