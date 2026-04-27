"""Phase 1 — Statistical Analysis"""
import sqlite3, os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
from paths import DB, OUT, data, plot_m, report, ensure_dirs
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (10, 6)
DPI = 150

def run():
    conn = sqlite3.connect(DB)

    # ── 1.1 Player career stats ──
    print("Step 1.1 — Player career stats")
    career = pd.read_sql("SELECT player_nick, player_name, team, total_score, total_games, efficiency AS career_efficiency FROM v_player_overall ORDER BY career_efficiency DESC", conn)
    pms = pd.read_sql("SELECT player_nick, efficiency FROM v_player_match_summary", conn)
    var_stats = pms.groupby('player_nick')['efficiency'].agg(['std','min','max']).reset_index()
    var_stats.columns = ['player_nick','eff_std','eff_min','eff_max']
    var_stats['eff_range'] = var_stats['eff_max'] - var_stats['eff_min']
    career = career.merge(var_stats, on='player_nick', how='left')
    career.to_csv(data("player_career_stats.csv"), index=False)
    print(f"  Saved {len(career)} rows")

    # ── 1.2 Team match efficiency ──
    print("Step 1.2 — Team match efficiency")
    tms = pd.read_sql("""
        SELECT pms.tournament_round, pms.match_id, pms.team,
            ROUND(SUM(pms.total_score)*1.0/SUM(pms.total_games),4) AS team_efficiency,
            CASE WHEN vm.team_a=pms.team THEN vm.score_a ELSE vm.score_b END AS team_pts,
            CASE WHEN vm.team_a=pms.team THEN vm.score_b ELSE vm.score_a END AS opp_pts,
            CASE WHEN (vm.team_a=pms.team AND vm.winner='A') OR (vm.team_b=pms.team AND vm.winner='B') THEN 1 ELSE 0 END AS match_won
        FROM v_player_match_summary pms
        JOIN v_matches vm ON vm.match_id=pms.match_id
        GROUP BY pms.tournament_round, pms.match_id, pms.team
    """, conn)
    tms['score_margin'] = tms['team_pts'] - tms['opp_pts']
    tms.to_csv(data("team_match_stats.csv"), index=False)
    print(f"  Saved {len(tms)} rows")

    # ── 1.3 Correlation ──
    print("Step 1.3 — Correlation: individual → team")
    pms_full = pd.read_sql("SELECT match_id, player_nick, team, efficiency FROM v_player_match_summary", conn)
    team_agg = pms_full.groupby(['match_id','team'])['efficiency'].agg(['mean','std','max','min']).reset_index()
    team_agg.columns = ['match_id','team','mean_player_eff','std_player_eff','max_player_eff','min_player_eff']
    corr_df = tms.merge(team_agg, on=['match_id','team'], how='inner')

    pairs = [('mean_player_eff','team_efficiency'), ('mean_player_eff','match_won'), ('std_player_eff','team_efficiency')]
    for x_col, y_col in pairs:
        valid = corr_df[[x_col, y_col]].dropna()
        if len(valid) > 2:
            pr, pp = stats.pearsonr(valid[x_col], valid[y_col])
            sr, sp = stats.spearmanr(valid[x_col], valid[y_col])
            print(f"  {x_col} vs {y_col}: Pearson r={pr:.3f} (p={pp:.4f}), Spearman ρ={sr:.3f} (p={sp:.4f})")

    # ── 1.4 Board contribution (OLS) ──
    print("Step 1.4 — Board contribution regression")
    pms_with_career = pms_full.merge(career[['player_nick','career_efficiency']], on='player_nick')
    pms_with_career['board_rank'] = pms_with_career.groupby(['match_id','team'])['career_efficiency'].rank(ascending=False, method='first').astype(int)
    board_wide = pms_with_career[pms_with_career['board_rank'] <= 3].pivot_table(
        index=['match_id','team'], columns='board_rank', values='efficiency', aggfunc='first'
    ).reset_index()
    board_wide.columns = ['match_id','team','board1_eff','board2_eff','board3_eff']
    board_df = board_wide.merge(tms[['match_id','team','score_margin']], on=['match_id','team'])
    board_df = board_df.dropna()

    if len(board_df) > 5:
        model = smf.ols("score_margin ~ board1_eff + board2_eff + board3_eff", data=board_df).fit()
        with open(report("board_contribution_regression.txt"), "w") as f:
            f.write(str(model.summary()))
        print(model.summary())

        # ── 1.5 Permutation test ──
        print("\nStep 1.5 — Permutation test (5000 iterations)")
        observed_betas = model.params[1:]
        perm_betas = {c: [] for c in observed_betas.index}
        for _ in range(5000):
            df_p = board_df.copy()
            df_p["score_margin"] = np.random.permutation(df_p["score_margin"])
            mp = smf.ols("score_margin ~ board1_eff + board2_eff + board3_eff", data=df_p).fit()
            for c in perm_betas:
                perm_betas[c].append(mp.params[c])
        for c in perm_betas:
            emp_p = np.mean(np.abs(perm_betas[c]) >= np.abs(observed_betas[c]))
            print(f"  {c}: β={observed_betas[c]:.3f}, permutation p={emp_p:.4f}")
    else:
        print("  ⚠ Not enough data for regression")

    # ── 1.6 Visualizations ──
    print("\nStep 1.6 — Visualizations")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(career['career_efficiency'], bins=20, edgecolor='black', alpha=0.7, color='#4C72B0')
    ax.set_xlabel('Career Efficiency'); ax.set_ylabel('Count'); ax.set_title('Distribution of Player Career Efficiency')
    plt.tight_layout(); fig.savefig(plot_m("hist_efficiency.png"), dpi=DPI); plt.close()

    if len(board_df) > 0:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for b, c in [(1,'#E24A33'),(2,'#348ABD'),(3,'#988ED5')]:
            col = f'board{b}_eff'
            ax.scatter(board_df[col], board_df['score_margin'], label=f'Board {b}', alpha=0.7, color=c, s=60)
        ax.set_xlabel('Board Efficiency'); ax.set_ylabel('Score Margin'); ax.set_title('Board Efficiency vs Score Margin')
        ax.legend(); plt.tight_layout(); fig.savefig(plot_m("scatter_board_contribution.png"), dpi=DPI); plt.close()

    num_cols = career.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(career[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap — Player Features')
    plt.tight_layout(); fig.savefig(plot_m("heatmap_corr.png"), dpi=DPI); plt.close()
    print("  ✓ All Phase 1 plots saved")

    conn.close()
    return career, tms, board_df

if __name__ == "__main__":
    run()
