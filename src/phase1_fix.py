"""
Phase 1 Fix Supplement
======================
Fixes: (1) circular dependency — use binary match_won instead of score_margin
       (2) outlier sensitivity check

Run AFTER phase1_stats.py has completed.

Note: uses sklearn LogisticRegression (L2 penalty) instead of statsmodels
logit because the data exhibits quasi-complete separation — board
efficiencies almost perfectly predict match_won, causing MLE to diverge.
"""
import sqlite3, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats

np.random.seed(42)
DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'twbc.db')
OUT = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

BOARD_COLS = ["board1_eff", "board2_eff", "board3_eff"]


def fit_logistic(df, feature_cols=BOARD_COLS, target="match_won", C=1.0):
    """Fit L2-regularized logistic regression. Returns model, coefs dict, accuracy, pseudo_r2."""
    X = df[feature_cols].values
    y = df[target].values
    model = LogisticRegression(C=C, l1_ratio=0, solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    coefs = dict(zip(feature_cols, model.coef_[0]))
    coefs["Intercept"] = model.intercept_[0]
    odds_ratios = {k: np.exp(v) for k, v in coefs.items()}
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    # McFadden pseudo-R²
    p_pred = model.predict_proba(X)[:, 1]
    eps = 1e-15
    ll_model = np.sum(y * np.log(p_pred + eps) + (1 - y) * np.log(1 - p_pred + eps))
    p0 = y.mean()
    ll_null = np.sum(y * np.log(p0 + eps) + (1 - y) * np.log(1 - p0 + eps))
    pseudo_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0
    return model, coefs, odds_ratios, accuracy, pseudo_r2


def run():
    con = sqlite3.connect(DB)

    # ══════════════════════════════════════════════════════
    # ORIGINAL PHASE 1 STEPS (pipeline prerequisites)
    # ══════════════════════════════════════════════════════

    # ── Step 0.1 — Player career stats ──
    print("Step 0.1 — Player career stats")
    career = pd.read_sql(
        "SELECT player_nick, player_name, team, total_score, total_games, "
        "efficiency AS career_efficiency FROM v_player_overall ORDER BY career_efficiency DESC",
        con
    )
    pms_var = pd.read_sql("SELECT player_nick, efficiency FROM v_player_match_summary", con)
    var_stats = pms_var.groupby('player_nick')['efficiency'].agg(['std','min','max']).reset_index()
    var_stats.columns = ['player_nick','eff_std','eff_min','eff_max']
    var_stats['eff_range'] = var_stats['eff_max'] - var_stats['eff_min']
    career = career.merge(var_stats, on='player_nick', how='left')
    career.to_csv(f"{OUT}/player_career_stats.csv", index=False)
    print(f"  Saved {len(career)} rows → player_career_stats.csv")

    # ── Step 0.2 — Team match stats ──
    print("Step 0.2 — Team match stats")
    tms = pd.read_sql("""
        SELECT pms.tournament_round, pms.match_id, pms.team,
            ROUND(SUM(pms.total_score)*1.0/SUM(pms.total_games),4) AS team_efficiency,
            CASE WHEN vm.team_a=pms.team THEN vm.score_a ELSE vm.score_b END AS team_pts,
            CASE WHEN vm.team_a=pms.team THEN vm.score_b ELSE vm.score_a END AS opp_pts,
            CASE WHEN (vm.team_a=pms.team AND vm.winner='A')
                   OR (vm.team_b=pms.team AND vm.winner='B') THEN 1 ELSE 0 END AS match_won
        FROM v_player_match_summary pms
        JOIN v_matches vm ON vm.match_id=pms.match_id
        GROUP BY pms.tournament_round, pms.match_id, pms.team
    """, con)
    tms['score_margin'] = tms['team_pts'] - tms['opp_pts']
    tms.to_csv(f"{OUT}/team_match_stats.csv", index=False)
    print(f"  Saved {len(tms)} rows → team_match_stats.csv")

    # ── Step 0.3 — Correlations ──
    print("Step 0.3 — Correlations: individual → team")
    pms_full = pd.read_sql(
        "SELECT match_id, player_nick, team, efficiency FROM v_player_match_summary", con
    )
    team_agg = pms_full.groupby(['match_id','team'])['efficiency'].agg(
        ['mean','std','max','min']
    ).reset_index()
    team_agg.columns = ['match_id','team','mean_player_eff','std_player_eff','max_player_eff','min_player_eff']
    corr_df = tms.merge(team_agg, on=['match_id','team'], how='inner')
    pairs = [
        ('mean_player_eff','team_efficiency'),
        ('mean_player_eff','match_won'),
        ('std_player_eff','team_efficiency')
    ]
    for x_col, y_col in pairs:
        valid = corr_df[[x_col, y_col]].dropna()
        if len(valid) > 2:
            pr, pp = scipy_stats.pearsonr(valid[x_col], valid[y_col])
            sr, sp = scipy_stats.spearmanr(valid[x_col], valid[y_col])
            print(f"  {x_col} vs {y_col}: Pearson r={pr:.3f} (p={pp:.4f}), Spearman ρ={sr:.3f} (p={sp:.4f})")

    # ── Step 0.4 — Visualizations ──
    print("Step 0.4 — Base visualizations")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(career['career_efficiency'], bins=20, edgecolor='black', alpha=0.7, color='#4C72B0')
    ax.set_xlabel('Career Efficiency'); ax.set_ylabel('Count')
    ax.set_title('Distribution of Player Career Efficiency')
    plt.tight_layout(); fig.savefig(f"{OUT}/hist_efficiency.png", dpi=DPI); plt.close()

    num_cols = career.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(career[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap — Player Features')
    plt.tight_layout(); fig.savefig(f"{OUT}/heatmap_corr.png", dpi=DPI); plt.close()
    print("  Saved: hist_efficiency.png, heatmap_corr.png")

    # ══════════════════════════════════════════════════════
    # FIX 1 — Remove circular dependency
    # ══════════════════════════════════════════════════════

    # ── Step 1.1 — Rebuild the regression dataset ──
    print("Step 1.1 — Rebuild regression dataset (binary target)")

    df_pms = pd.read_sql("""
        SELECT pms.tournament_round, pms.match_id, pms.player_nick,
               pms.team, pms.total_score, pms.total_games, pms.efficiency
        FROM v_player_match_summary pms
        ORDER BY pms.match_id, pms.team, pms.efficiency DESC
    """, con)

    df_overall = pd.read_sql(
        "SELECT player_nick, efficiency AS career_eff FROM v_player_overall", con
    )
    df_pms = df_pms.merge(df_overall, on="player_nick")

    df_matches = pd.read_sql(
        "SELECT match_id, team_a, team_b, winner FROM v_matches", con
    )

    # Assign board positions by career_eff rank within (match, team)
    df_pms["board"] = (
        df_pms.groupby(["match_id", "team"])["career_eff"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Keep only boards 1-3
    df_boards = df_pms[df_pms.board <= 3].copy()

    # Pivot to wide format
    df_wide = df_boards.pivot_table(
        index=["match_id", "team"],
        columns="board",
        values="efficiency"
    ).reset_index()
    df_wide.columns = ["match_id", "team", "board1_eff", "board2_eff", "board3_eff"]

    # Add binary match outcome
    df_wide = df_wide.merge(df_matches, on="match_id")
    df_wide["match_won"] = (
        ((df_wide.team == df_wide.team_a) & (df_wide.winner == "A")) |
        ((df_wide.team == df_wide.team_b) & (df_wide.winner == "B"))
    ).astype(int)

    # Add team points for visualization
    df_score = pd.read_sql("""
        SELECT pms.match_id, pms.team, SUM(pms.total_score) AS team_pts
        FROM v_player_match_summary pms
        GROUP BY pms.match_id, pms.team
    """, con)
    df_wide = df_wide.merge(df_score, on=["match_id", "team"])

    print(f"  Dataset: {len(df_wide)} rows")
    print(df_wide[["team", "board1_eff", "board2_eff", "board3_eff", "match_won"]].head(6).to_string(index=False))

    # ── Step 1.2 — Logistic regression (L2-regularized) ──
    print("\nStep 1.2 — L2-regularized logistic regression")
    print("  (statsmodels MLE diverges due to quasi-complete separation)")

    df_fit = df_wide.dropna().copy()
    logit_model, coefs, odds_ratios, accuracy, pseudo_r2 = fit_logistic(df_fit)

    # Build a summary report
    summary_lines = [
        "Penalized Logistic Regression (L2, C=1.0)",
        "=" * 50,
        f"Observations: {len(df_fit)}",
        f"Target: match_won (binary)",
        "",
        f"{'Feature':<15} {'Coef':>10} {'Odds Ratio':>12}",
        "-" * 40,
    ]
    for col in BOARD_COLS:
        summary_lines.append(f"{col:<15} {coefs[col]:>10.4f} {odds_ratios[col]:>12.3f}")
    summary_lines.append(f"{'Intercept':<15} {coefs['Intercept']:>10.4f} {odds_ratios['Intercept']:>12.3f}")
    summary_lines.extend(["", f"McFadden pseudo-R²: {pseudo_r2:.4f}", f"Classification accuracy: {accuracy:.1%}"])
    summary_text = "\n".join(summary_lines)

    print(summary_text)
    with open(f"{OUT}/board_contribution_logistic.txt", "w") as f:
        f.write(summary_text)
    print("  Saved: output/board_contribution_logistic.txt")

    # ── Step 1.3 — Permutation test ──
    print("\nStep 1.3 — Permutation test (2000 iterations)")

    observed_coef = {c: coefs[c] for c in BOARD_COLS}
    N_PERM = 2000
    perm_coefs = {col: [] for col in BOARD_COLS}

    for i in range(N_PERM):
        df_p = df_fit.copy()
        df_p["match_won"] = np.random.permutation(df_p["match_won"])
        try:
            _, pc, _, _, _ = fit_logistic(df_p)
            for col in BOARD_COLS:
                perm_coefs[col].append(pc[col])
        except Exception:
            pass

    print("  Permutation test results:")
    for col in BOARD_COLS:
        obs = observed_coef[col]
        emp_p = np.mean(np.abs(perm_coefs[col]) >= np.abs(obs))
        print(f"    {col}: coef={obs:.3f}, permutation p={emp_p:.4f}")

    # ══════════════════════════════════════════════════════
    # FIX 2 — Outlier sensitivity check
    # ══════════════════════════════════════════════════════

    # ── Step 2.1 — Identify outlier players ──
    print("\n" + "─" * 50)
    print("Step 2.1 — Identify outlier players")

    df_all_players = pd.read_sql(
        "SELECT player_nick, player_name, team, efficiency FROM v_player_overall", con
    )
    OUTLIER_THRESHOLD = 0.10
    outliers = df_all_players[df_all_players.efficiency < OUTLIER_THRESHOLD]
    print(f"  Outlier players (efficiency < {OUTLIER_THRESHOLD}):")
    print(outliers[["player_nick", "player_name", "team", "efficiency"]].to_string(index=False))

    outlier_nicks = set(outliers.player_nick)
    outlier_matches = df_pms[df_pms.player_nick.isin(outlier_nicks)]["match_id"].unique()
    print(f"\n  Matches containing outlier players: {len(outlier_matches)}")

    # ── Step 2.2 — Refit without outlier matches ──
    print("\nStep 2.2 — Refit with and without outlier matches")

    df_clean = df_wide[~df_wide.match_id.isin(outlier_matches)].dropna()
    print(f"  Full dataset: {len(df_fit)} rows")
    print(f"  Clean dataset (no outlier matches): {len(df_clean)} rows")
    print(f"  Removed: {len(df_fit) - len(df_clean)} rows")

    _, coefs_clean, _, _, _ = fit_logistic(df_clean)

    comparison = pd.DataFrame({
        "coef_full": pd.Series({c: coefs[c] for c in BOARD_COLS}),
        "coef_clean": pd.Series({c: coefs_clean[c] for c in BOARD_COLS}),
    })
    comparison["delta"] = comparison.coef_clean - comparison.coef_full
    comparison["pct_change"] = (comparison.delta / comparison.coef_full.abs() * 100).round(1)

    print("\n  === Coefficient stability check ===")
    print(comparison.round(4).to_string())
    print("""
  Interpretation:
    |pct_change| < 10%  → STABLE — outliers not driving result
    |pct_change| 10-30% → MODERATE — report both models
    |pct_change| > 30%  → DRIVEN BY OUTLIERS — use clean model
    """)

    comparison.to_csv(f"{OUT}/board_coef_stability.csv")
    print("  Saved: output/board_coef_stability.csv")

    # ── Step 2.3 — Visualize outlier impact ──
    print("Step 2.3 — Visualize outlier impact")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors_map = {"board1_eff": "#E24B4A", "board2_eff": "#378ADD", "board3_eff": "#7F77DD"}
    for b in BOARD_COLS:
        mask_out = df_wide["match_id"].isin(outlier_matches)
        ax.scatter(df_wide.loc[~mask_out, b], df_wide.loc[~mask_out, "team_pts"],
                   color=colors_map[b], alpha=0.5, s=40, label=b.replace("_eff", ""))
        ax.scatter(df_wide.loc[mask_out, b], df_wide.loc[mask_out, "team_pts"],
                   color=colors_map[b], alpha=0.9, s=80, marker="x", linewidths=1.5)
    outlier_patch = plt.Line2D([0], [0], marker="x", color="gray", linestyle="None",
                               markersize=8, label="contains outlier player")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:3] + [outlier_patch], fontsize=9)
    ax.set_xlabel("Board Efficiency"); ax.set_ylabel("Team Points Scored")
    ax.set_title("Outlier matches marked with ×")

    ax2 = axes[1]
    x = np.arange(3); w = 0.35
    full_vals = [coefs[c] for c in BOARD_COLS]
    clean_vals = [coefs_clean[c] for c in BOARD_COLS]
    ax2.bar(x - w/2, full_vals, w, label="Full dataset", color="#378ADD", alpha=0.8)
    ax2.bar(x + w/2, clean_vals, w, label="Outliers removed", color="#1D9E75", alpha=0.8)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(["Board 1", "Board 2", "Board 3"])
    ax2.set_ylabel("Logistic coefficient (log-odds)")
    ax2.set_title("Coefficient stability: full vs clean dataset")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{OUT}/outlier_sensitivity.png", dpi=DPI); plt.close()
    print("  Saved: output/outlier_sensitivity.png")

    # ══════════════════════════════════════════════════════
    # Step 3 — Revised summary
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  PHASE 1 REVISED SUMMARY")
    print("=" * 55)

    print("""
OLD (OLS on score_margin):
  R² = 0.997  ← circular dependency, not meaningful
  β ≈ 71 for all boards  ← tautological result
  JB test failed (skew=-3.6, kurt=20.9)

NEW (L2-Regularized Logistic on match_won):""")

    for col in BOARD_COLS:
        print(f"  {col}: log-odds={coefs[col]:.3f}, OR={odds_ratios[col]:.2f}")

    print(f"""
  McFadden pseudo-R²: {pseudo_r2:.4f}
  Accuracy: {accuracy:.1%}

OUTLIER SENSITIVITY:""")

    for idx, row in comparison.iterrows():
        flag = ("STABLE" if abs(row['pct_change']) < 10
                else ("SENSITIVE" if abs(row['pct_change']) < 30
                      else "DRIVEN BY OUTLIERS"))
        print(f"  {idx}: Δ={row['delta']:+.3f} ({row['pct_change']:+.1f}%)  → {flag}")

    # Output checklist
    print("\n" + "─" * 40)
    expected = [
        "board_contribution_logistic.txt",
        "board_coef_stability.csv",
        "outlier_sensitivity.png",
    ]
    for f in expected:
        exists = os.path.isfile(os.path.join(OUT, f))
        mark = "✓" if exists else "✗"
        print(f"  [{mark}] output/{f}")

    con.close()
    return career, tms, coefs


if __name__ == "__main__":
    run()
