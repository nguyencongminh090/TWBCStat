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
from scipy.optimize import linear_sum_assignment
from paths import DB, data, TRENDS_DIR, ensure_dirs

np.random.seed(42)
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


# ─────────────────────────────────────────────────────────────
# Step 1 — Load prerequisites
# ─────────────────────────────────────────────────────────────

def _load():
    elo_df = pd.read_csv(data("elo_ratings_final.csv"))
    elo = dict(zip(elo_df.nick, elo_df.elo))

    bayes = pd.read_csv(data("bayesian_head_to_head.csv"))
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
    win_trends.to_csv(data("win_trends_per_player.csv"), index=False)
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
    divergence.to_csv(data("top_divergences.csv"), index=False)
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
# Step 6 — Optimal lineup (Hungarian algorithm)
# ─────────────────────────────────────────────────────────────

def optimal_lineup(team_a: list, team_b: list, elo_dict: dict,
                   n_games: int = 12, team_a_name: str = "Team A",
                   team_b_name: str = "Team B") -> pd.DataFrame:
    """
    Find the optimal board assignment for Team A using the
    Hungarian algorithm (linear_sum_assignment).

    Solves:  maximize  Σ_i  P(team_a[i] beats team_b[σ(i)])
    over all permutations σ of team_b players.

    Also finds the WORST-case assignment (what Team B wants).

    Parameters
    ----------
    team_a, team_b : lists of player nicks (available roster, not yet assigned)
    elo_dict       : nick → Elo rating
    n_games        : games per board

    Returns
    -------
    DataFrame comparing: given order, best-for-A, worst-for-A
    """
    na, nb = len(team_a), len(team_b)
    n = min(na, nb)

    # Build cost matrix: -p because linear_sum_assignment MINIMIZES
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ea = elo_dict.get(team_a[i], 1200)
            eb = elo_dict.get(team_b[j], 1200)
            cost[i, j] = -(1 / (1 + 10 ** ((eb - ea) / 400)))

    # Best assignment for Team A (maximize total p)
    row_best, col_best = linear_sum_assignment(cost)
    best_pairs = [(team_a[r], team_b[c]) for r, c in zip(row_best, col_best)]
    best_total_p = -cost[row_best, col_best].sum()

    # Worst assignment for Team A (minimize total p)
    row_worst, col_worst = linear_sum_assignment(-cost)
    worst_pairs = [(team_a[r], team_b[c]) for r, c in zip(row_worst, col_worst)]
    worst_total_p = -cost[row_worst, col_worst].sum()  # = sum of p values

    # Default assignment (as given)
    default_p = sum(
        1 / (1 + 10 ** ((elo_dict.get(team_b[i], 1200) -
                         elo_dict.get(team_a[i], 1200)) / 400))
        for i in range(n)
    )

    # Print comparison
    print(f"\n  Optimal Lineup Analysis: {team_a_name} vs {team_b_name}")
    print(f"  {'─'*65}")

    def _print_assignment(label, pairs, total_p):
        print(f"\n  {label} (total P = {total_p:.4f}, E[score_A] = {total_p*n_games:.1f}):")
        for i, (a, b) in enumerate(pairs):
            ea = elo_dict.get(a, 1200)
            eb = elo_dict.get(b, 1200)
            p = 1 / (1 + 10 ** ((eb - ea) / 400))
            print(f"    Board {i+1}: {a:<16} vs {b:<16} P(A)={p:.3f}  E[A]={p*n_games:.1f}")

    _print_assignment("Given order",
                      list(zip(team_a[:n], team_b[:n])), default_p)
    _print_assignment(f"Best for {team_a_name} (Hungarian)",
                      best_pairs, best_total_p)
    _print_assignment(f"Worst for {team_a_name}",
                      worst_pairs, worst_total_p)

    swing = (best_total_p - worst_total_p) * n_games
    print(f"\n  Swing range: {swing:.1f} expected points between best and worst assignment")
    print(f"  Improvement over given: {(best_total_p - default_p)*n_games:+.1f} points")

    # Build output DataFrame
    rows = []
    for i, (a, b) in enumerate(best_pairs):
        ea = elo_dict.get(a, 1200)
        eb = elo_dict.get(b, 1200)
        p = 1 / (1 + 10 ** ((eb - ea) / 400))
        rows.append({"board": i+1, "player_a": a, "player_b": b,
                     "p_a_wins": round(p, 4),
                     "expected_a": round(p * n_games, 2),
                     "assignment": "optimal"})
    return pd.DataFrame(rows)


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

    print("\n" + "─" * 55)
    print("  Step 6 — Optimal lineup (Hungarian algorithm)")
    print("─" * 55)
    optimal_lineup(
        team_a=["wbcbeetle", "wbcbb", "wbcpuholek"],
        team_b=["wbcd",      "wbca",  "wbcc"],
        elo_dict=elo,
        team_a_name="Poland A", team_b_name="Czechia A",
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
