"""Phase 3 — Adjusted Metrics, Composite Index, Validation"""
import sqlite3, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from phase2_elo import compute_elo

np.random.seed(42)
from paths import DB, OUT, data, plot_m, ensure_dirs
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

def run():
    conn = sqlite3.connect(DB)

    career = pd.read_csv(data("player_career_stats.csv"))
    elo_df = pd.read_csv(data("elo_ratings_final.csv"))

    # ── 3.1 Adjusted Efficiency ──
    print("Step 3.1 — Adjusted Efficiency")
    G_max = career['total_games'].max()
    career['adjusted_efficiency'] = career['career_efficiency'] * np.log(career['total_games']) / np.log(G_max)

    # ── 3.2 Strength of Schedule ──
    print("Step 3.2 — Strength of Schedule")
    opp = pd.read_sql("""
        SELECT pa.nick AS player, pb.nick AS opponent FROM v_pairings vp
        JOIN players pa ON pa.player_id=vp.player_a_id JOIN players pb ON pb.player_id=vp.player_b_id
        UNION ALL
        SELECT pb.nick AS player, pa.nick AS opponent FROM v_pairings vp
        JOIN players pa ON pa.player_id=vp.player_a_id JOIN players pb ON pb.player_id=vp.player_b_id
    """, conn)
    opp_eff = opp.merge(career[['player_nick','career_efficiency']].rename(columns={'player_nick':'opponent','career_efficiency':'opp_eff'}), on='opponent')
    sos = opp_eff.groupby('player')['opp_eff'].mean().reset_index()
    sos.columns = ['player_nick', 'sos']
    career = career.merge(sos, on='player_nick', how='left')

    # ── 3.3 Composite Index ──
    print("Step 3.3 — Composite Performance Index")
    career['consistency'] = 1 - career['eff_range'].fillna(0)
    career['rank'] = career['career_efficiency'].rank(ascending=False)
    features = ['adjusted_efficiency', 'sos', 'consistency']
    valid = career.dropna(subset=features)

    scaler = StandardScaler()
    X = scaler.fit_transform(valid[features])
    y = -valid['rank'].values

    reg = LinearRegression().fit(X, y)
    print(f"  Weights: {dict(zip(features, reg.coef_.round(3)))}")
    print(f"  R²: {reg.score(X, y):.4f}")
    valid = valid.copy()
    valid['composite_index'] = reg.predict(X)
    valid.sort_values('composite_index', ascending=False).to_csv(f"{OUT}/composite_index.csv", index=False)

    # ── 3.4 Validate on Round 3 ──
    print("\nStep 3.4 — Validate on Round 3")
    pairings_all = pd.read_sql("""
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
    career_games = dict(zip(career['player_nick'], career['total_games']))
    elo_r2, _ = compute_elo(pairings_all, career_games, max_round=2)

    r3_matches = pd.read_sql("""
        SELECT m.match_id, vm.team_a, vm.team_b, vm.winner AS actual_winner
        FROM matches m JOIN v_matches vm ON vm.match_id=m.match_id
        WHERE m.tournament_round=3
    """, conn)
    r3_pairings = pd.read_sql("""
        SELECT m.match_id, vm.team_a, vm.team_b, pa.nick AS nick_a, pb.nick AS nick_b,
            vp.total_games, vp.winner AS actual_pairing_winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id=vp.sub_round_id
        JOIN matches m ON m.match_id=sr.match_id
        JOIN v_matches vm ON vm.match_id=m.match_id
        JOIN players pa ON pa.player_id=vp.player_a_id
        JOIN players pb ON pb.player_id=vp.player_b_id
        WHERE m.tournament_round=3
    """, conn)

    # Predict per pairing
    r3_pairings['p_a'] = r3_pairings.apply(
        lambda r: 1/(1+10**((elo_r2.get(r.nick_b,1200)-elo_r2.get(r.nick_a,1200))/400)), axis=1)
    r3_pairings['exp_score_a'] = r3_pairings['p_a'] * r3_pairings['total_games']
    r3_pairings['exp_score_b'] = (1 - r3_pairings['p_a']) * r3_pairings['total_games']

    # Aggregate per match
    pred = r3_pairings.groupby('match_id').agg(
        team_a=('team_a','first'), team_b=('team_b','first'),
        pred_score_a=('exp_score_a','sum'), pred_score_b=('exp_score_b','sum')
    ).reset_index()
    pred['predicted_winner'] = np.where(pred['pred_score_a'] > pred['pred_score_b'], 'A', 'B')
    pred = pred.merge(r3_matches[['match_id','actual_winner']], on='match_id')
    pred['correct'] = pred['predicted_winner'] == pred['actual_winner']

    accuracy = pred['correct'].mean()
    # Brier score
    pred['p_a_win'] = pred['pred_score_a'] / (pred['pred_score_a'] + pred['pred_score_b'])
    pred['outcome'] = (pred['actual_winner'] == 'A').astype(float)
    brier = ((pred['p_a_win'] - pred['outcome'])**2).mean()

    print(f"\n  Match-level accuracy: {accuracy:.2%} ({pred['correct'].sum()}/{len(pred)})")
    print(f"  Brier score: {brier:.4f}")
    print(f"\n  {'Match':<45} {'Predicted':>10} {'Actual':>8} {'✓/✗':>4}")
    print(f"  {'─'*45} {'─'*10} {'─'*8} {'─'*4}")
    for _, r in pred.iterrows():
        mark = '✓' if r['correct'] else '✗'
        print(f"  {r.team_a+' vs '+r.team_b:<45} {r.predicted_winner:>10} {r.actual_winner:>8} {mark:>4}")
    pred.to_csv(f"{OUT}/validation_round3.csv", index=False)

    # ── 3.5 Visualizations ──
    print("\nStep 3.5 — Final visualizations")

    # Adjusted efficiency scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid['career_efficiency'], valid['adjusted_efficiency'], alpha=0.6, s=50, color='#4C72B0')
    top5 = valid.nlargest(5, 'adjusted_efficiency')
    bot5 = valid.nsmallest(5, 'adjusted_efficiency')
    for _, r in pd.concat([top5, bot5]).iterrows():
        ax.annotate(r['player_nick'], (r['career_efficiency'], r['adjusted_efficiency']),
                    fontsize=7, ha='left', va='bottom')
    ax.set_xlabel('Career Efficiency'); ax.set_ylabel('Adjusted Efficiency')
    ax.set_title('Career vs Adjusted Efficiency'); ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5)
    plt.tight_layout(); fig.savefig(f"{OUT}/scatter_adjusted_efficiency.png", dpi=DPI); plt.close()

    # Composite index bar
    top20 = valid.nlargest(20, 'composite_index')
    teams_u = top20['team'].unique()
    colors = dict(zip(teams_u, sns.color_palette('husl', len(teams_u))))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top20)), top20['composite_index'], color=[colors[t] for t in top20['team']])
    ax.set_yticks(range(len(top20))); ax.set_yticklabels([f"{r.player_nick} ({r.team})" for _, r in top20.iterrows()], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('Composite Index'); ax.set_title('Top 20 Players — Composite Performance Index')
    plt.tight_layout(); fig.savefig(f"{OUT}/bar_composite_index.png", dpi=DPI); plt.close()

    # Validation table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    tbl = ax.table(cellText=pred[['team_a','team_b','predicted_winner','actual_winner','correct']].values,
                   colLabels=['Team A','Team B','Predicted','Actual','Correct'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.5)
    ax.set_title('Round 3 Predictions vs Actuals', pad=20)
    plt.tight_layout(); fig.savefig(f"{OUT}/table_validation.png", dpi=DPI); plt.close()
    print("  ✓ All Phase 3 plots saved")

    conn.close()
    return valid, pred, accuracy, brier

if __name__ == "__main__":
    run()
