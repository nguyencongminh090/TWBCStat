"""
Supplementary Analysis: Player Win Trend vs Opponents
=====================================================
Builds on top of Phase 2 outputs to answer:
  "For each player, how does their predicted win probability shift
   depending on which specific opponent they face — and how confident
   are we in that prediction?"

Prerequisite outputs:
  - output/elo_ratings_final.csv
  - output/bayesian_head_to_head.csv
"""
import sqlite3, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta as beta_dist

np.random.seed(42)

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(BASE, "output")
TRENDS_DIR = os.path.join(OUT, "trends")
os.makedirs(TRENDS_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


# ─────────────────────────────────────────────────────────────
# Step 1 — Load prerequisites
# ─────────────────────────────────────────────────────────────

def _load():
    elo_df = pd.read_csv(f"{OUT}/elo_ratings_final.csv")
    elo = dict(zip(elo_df.nick, elo_df.elo))

    bayes = pd.read_csv(f"{OUT}/bayesian_head_to_head.csv")
    print(f"Loaded Elo ratings for {len(elo)} players")
    print(f"Loaded Bayesian h2h: {len(bayes)} matchups")
    return elo, bayes


# ─────────────────────────────────────────────────────────────
# Step 2 — Build unified win probability table
# ─────────────────────────────────────────────────────────────

def _build_win_trends(elo, bayes):
    ALPHA_PRIOR, BETA_PRIOR = 2, 2

    records = []
    for _, row in bayes.iterrows():
        # Side A perspective
        a_post  = ALPHA_PRIOR + row.a_wins
        b_post  = BETA_PRIOR  + row.b_wins
        p_bayes = a_post / (a_post + b_post)
        ci_lo   = beta_dist.ppf(0.025, a_post, b_post)
        ci_hi   = beta_dist.ppf(0.975, a_post, b_post)
        p_elo   = 1 / (1 + 10 ** ((elo.get(row.nick_b, 1200) -
                                    elo.get(row.nick_a, 1200)) / 400))

        records.append({
            "player":      row.nick_a,
            "opponent":    row.nick_b,
            "games":       row.games,
            "p_elo":       round(p_elo, 4),
            "p_bayes":     round(p_bayes, 4),
            "ci_lower":    round(ci_lo, 4),
            "ci_upper":    round(ci_hi, 4),
            "uncertainty": round(ci_hi - ci_lo, 4),
            "signal":      round(p_bayes - p_elo, 4),
        })

        # Reverse direction (B's perspective)
        records.append({
            "player":      row.nick_b,
            "opponent":    row.nick_a,
            "games":       row.games,
            "p_elo":       round(1 - p_elo, 4),
            "p_bayes":     round(1 - p_bayes, 4),
            "ci_lower":    round(1 - ci_hi, 4),
            "ci_upper":    round(1 - ci_lo, 4),
            "uncertainty": round(ci_hi - ci_lo, 4),
            "signal":      round((1 - p_bayes) - (1 - p_elo), 4),
        })

    win_trends = pd.DataFrame(records).sort_values(
        ["player", "p_bayes"], ascending=[True, False]
    )
    win_trends.to_csv(f"{OUT}/win_trends_per_player.csv", index=False)
    print(f"Saved {len(win_trends)} rows to output/win_trends_per_player.csv")
    return win_trends


# ─────────────────────────────────────────────────────────────
# Step 3 — Per-player trend charts
# ─────────────────────────────────────────────────────────────

def _plot_trends(win_trends):
    players_to_plot = (
        win_trends.groupby("player")["opponent"]
        .nunique()
        .where(lambda x: x >= 2)
        .dropna()
        .index.tolist()
    )

    for player_nick in players_to_plot:
        df_p = win_trends[win_trends.player == player_nick].copy()
        df_p = df_p.sort_values("p_bayes", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(max(8, len(df_p) * 1.2), 5))
        x = np.arange(len(df_p))

        # Elo bars (background)
        ax.bar(x, df_p.p_elo, width=0.6, color="#90b4d4", alpha=0.6,
               label="Elo prediction")

        # Bayesian point + CI
        for i, row in df_p.iterrows():
            ls = "-" if row.uncertainty < 0.3 else "--"
            ax.plot([i, i], [row.ci_lower, row.ci_upper],
                    color="#e07b39", linewidth=2, linestyle=ls)
            ax.scatter(i, row.p_bayes, color="#e07b39", zorder=5, s=60)

        # Even odds line
        ax.axhline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.8)

        # X-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row.opponent}\n({int(row.games)}g)" for _, row in df_p.iterrows()],
            fontsize=9
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Win probability")
        ax.set_title(f"{player_nick} — win probability vs each opponent", fontsize=12)

        legend_elements = [
            mpatches.Patch(facecolor="#90b4d4", alpha=0.7, label="Elo prediction"),
            plt.Line2D([0], [0], color="#e07b39", marker="o", linestyle="-",
                       label="Bayesian estimate (95% CI)"),
            plt.Line2D([0], [0], color="#e07b39", linestyle="--",
                       label="Wide CI = few games"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

        plt.tight_layout()
        fname = os.path.join(TRENDS_DIR, f"{player_nick}_win_trend.png")
        plt.savefig(fname, dpi=DPI)
        plt.close()
        print(f"  Saved: {fname}")

    print(f"\nTotal charts generated: {len(players_to_plot)}")


# ─────────────────────────────────────────────────────────────
# Step 4 — Summary table: biggest Elo vs Bayesian divergences
# ─────────────────────────────────────────────────────────────

def _divergence_table(win_trends):
    # Try ≥3 games first; if too few rows, fall back to ≥2 then ≥1
    for min_games in [3, 2, 1]:
        divergence = win_trends[win_trends.games >= min_games].copy()
        if len(divergence) >= 5:
            break

    divergence["abs_signal"] = divergence.signal.abs()
    divergence = divergence.sort_values("abs_signal", ascending=False).head(20)

    print(f"\n=== Top 20 biggest Elo vs Bayesian divergences (min {min_games} game(s)) ===")
    print(divergence[["player","opponent","games","p_elo","p_bayes",
                       "signal","uncertainty"]].to_string(index=False))
    divergence.to_csv(f"{OUT}/top_divergences.csv", index=False)
    print(f"Saved: output/top_divergences.csv")

    print("""
Signal interpretation:
  signal > +0.15  →  player CONSISTENTLY beats this opponent more than Elo predicts
  signal < -0.15  →  player CONSISTENTLY loses to this opponent more than Elo predicts
  |signal| < 0.10 →  Elo and actual record agree — matchup is well-calibrated
""")
    return divergence


# ─────────────────────────────────────────────────────────────
# Step 5 — Team lineup simulator
# ─────────────────────────────────────────────────────────────

def predict_lineup(lineup_a: list, lineup_b: list, elo_dict: dict,
                   n_games: int = 12) -> pd.DataFrame:
    """
    Predict expected scores for a proposed team matchup.

    Parameters
    ----------
    lineup_a, lineup_b : lists of player_nick (same length, order = board pairing)
    elo_dict           : dict nick → Elo rating
    n_games            : games per board pairing (default 12)

    Returns
    -------
    DataFrame with per-board and total predictions
    """
    results = []
    for nick_a, nick_b in zip(lineup_a, lineup_b):
        p = 1 / (1 + 10 ** ((elo_dict.get(nick_b, 1200) -
                              elo_dict.get(nick_a, 1200)) / 400))
        results.append({
            "board":           f"Board {len(results)+1}",
            "player_a":        nick_a,
            "player_b":        nick_b,
            "p_a_wins":        round(p, 4),
            "expected_score_a": round(p * n_games, 2),
            "expected_score_b": round((1 - p) * n_games, 2),
        })

    df_r = pd.DataFrame(results)
    total_a = df_r.expected_score_a.sum()
    total_b = df_r.expected_score_b.sum()
    total_row = pd.DataFrame([{
        "board": "—TOTAL—",
        "player_a": "Team A",
        "player_b": "Team B",
        "p_a_wins": "—",
        "expected_score_a": round(total_a, 2),
        "expected_score_b": round(total_b, 2),
    }])
    df_out = pd.concat([df_r, total_row], ignore_index=True)

    print(df_out.to_string(index=False))
    winner = "Team A" if total_a > total_b else "Team B"
    margin = abs(total_a - total_b)
    print(f"\nPredicted winner: {winner}  (margin: {margin:.1f} points)")
    return df_out


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  Supplementary: Player Win Trends vs Opponents")
    print("=" * 60)

    elo, bayes = _load()

    print("\n" + "─" * 55)
    print("  Step 2 — Build unified win probability table")
    print("─" * 55)
    win_trends = _build_win_trends(elo, bayes)

    print("\n" + "─" * 55)
    print("  Step 3 — Per-player trend charts")
    print("─" * 55)
    _plot_trends(win_trends)

    print("\n" + "─" * 55)
    print("  Step 4 — Biggest Elo vs Bayesian divergences")
    print("─" * 55)
    _divergence_table(win_trends)

    print("\n" + "─" * 55)
    print("  Step 5 — Example lineup simulation")
    print("─" * 55)
    # Poland A vs Czechia A (top teams by individual quality)
    predict_lineup(
        lineup_a=["wbcbeetle", "wbcbb", "wbcpuholek"],
        lineup_b=["wbcd",      "wbca",  "wbcc"],
        elo_dict=elo,
    )

    # Output checklist
    print("\n" + "=" * 60)
    print("  OUTPUT CHECKLIST")
    print("=" * 60)
    checks = [
        "output/win_trends_per_player.csv",
        "output/top_divergences.csv",
    ]
    for f in checks:
        exists = os.path.isfile(os.path.join(BASE, f))
        mark = "✓" if exists else "✗"
        print(f"  [{mark}] {f}")

    # Count trend charts
    trend_files = [f for f in os.listdir(TRENDS_DIR) if f.endswith("_win_trend.png")]
    print(f"  [{'✓' if trend_files else '✗'}] output/trends/ — {len(trend_files)} charts")

    return win_trends


if __name__ == "__main__":
    run()
