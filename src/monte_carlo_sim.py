"""
Monte Carlo Match & Tournament Simulator — TWBC 2026
=====================================================
Applies Monte Carlo simulation to answer questions that point-estimate
models (Elo, Bayesian h2h) cannot:

  1. What is the DISTRIBUTION of possible match scores?
  2. What is the probability of an upset at team level?
  3. How do full-tournament standings shake out across 10,000 simulations?
  4. How sensitive are team standings to individual board matchups?

Theory:
  Monte Carlo simulation replaces analytical probability calculations
  with repeated random sampling. Instead of computing P(Team A wins)
  as a single number, we simulate 10,000 plausible tournaments and
  count how often each outcome occurs. This naturally captures:
    - Variance from small sample sizes
    - Interaction effects between boards
    - Fat tails and upset scenarios
    - Confidence intervals on any derived statistic

Prerequisites:
  - output/elo_ratings_final.csv
  - Database: data/processed/twbc.db
"""
import sqlite3, os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

np.random.seed(42)

BASE = os.path.join(os.path.dirname(__file__), "..")
DB   = os.path.join(BASE, "data", "processed", "twbc.db")
OUT  = os.path.join(BASE, "output")
MC_DIR = os.path.join(OUT, "monte_carlo")
os.makedirs(MC_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

N_SIMS = 10_000
GAMES_PER_BOARD = 12   # standard TWBC pairing length


# ═══════════════════════════════════════════════════════════════
# CORE ENGINE: Monte Carlo Board/Match/Tournament Simulator
# ═══════════════════════════════════════════════════════════════

def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Expected score for player A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def simulate_board(p_a: float, n_games: int, n_sims: int) -> np.ndarray:
    """
    Simulate n_sims instances of a board pairing.

    Each game is Bernoulli(p_a) → score_a counts wins for A.
    Returns array of shape (n_sims,) with score_a for each simulation.

    This is the atomic building block: each game is an independent
    coin flip weighted by Elo difference.
    """
    return np.random.binomial(n_games, p_a, size=n_sims)


def simulate_match(lineup_a: list, lineup_b: list,
                   elo_dict: dict, n_games: int = GAMES_PER_BOARD,
                   n_sims: int = N_SIMS) -> dict:
    """
    Simulate a full team match (3 boards, each playing n_games).

    Returns:
        dict with per-board and aggregate results across n_sims simulations.
    """
    board_results = []
    total_a = np.zeros(n_sims)
    total_b = np.zeros(n_sims)

    for i, (a, b) in enumerate(zip(lineup_a, lineup_b)):
        p_a = elo_win_prob(elo_dict.get(a, 1200), elo_dict.get(b, 1200))
        scores_a = simulate_board(p_a, n_games, n_sims)
        scores_b = n_games - scores_a

        total_a += scores_a
        total_b += scores_b

        board_results.append({
            "board":      i + 1,
            "player_a":   a,
            "player_b":   b,
            "p_a":        p_a,
            "mean_a":     scores_a.mean(),
            "mean_b":     scores_b.mean(),
            "std_a":      scores_a.std(),
            "scores_a":   scores_a,   # full distribution kept for plots
        })

    # Aggregate team-level outcomes
    a_wins  = (total_a > total_b).sum() / n_sims
    b_wins  = (total_b > total_a).sum() / n_sims
    draws   = (total_a == total_b).sum() / n_sims

    return {
        "boards":     board_results,
        "total_a":    total_a,
        "total_b":    total_b,
        "p_team_a":   a_wins,
        "p_team_b":   b_wins,
        "p_draw":     draws,
        "mean_margin": (total_a - total_b).mean(),
        "std_margin":  (total_a - total_b).std(),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Single Match Deep Dive
# ═══════════════════════════════════════════════════════════════

def analyze_match(lineup_a, lineup_b, elo_dict, team_a_name="Team A",
                  team_b_name="Team B", save=True):
    """Run MC simulation for one match, print analysis, optionally save plot."""
    result = simulate_match(lineup_a, lineup_b, elo_dict)
    margin = result["total_a"] - result["total_b"]

    print(f"\n{'─'*60}")
    print(f"  {team_a_name} vs {team_b_name}  ({N_SIMS:,} simulations)")
    print(f"{'─'*60}")
    print(f"\n  Board-level breakdown:")
    print(f"  {'Board':<8} {'Player A':<18} {'Player B':<18} {'p(A)':>6} {'E[A]':>6} {'E[B]':>6} {'σ':>5}")
    print(f"  {'─'*8} {'─'*18} {'─'*18} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
    for br in result["boards"]:
        print(f"  {br['board']:<8} {br['player_a']:<18} {br['player_b']:<18} "
              f"{br['p_a']:>6.3f} {br['mean_a']:>6.2f} {br['mean_b']:>6.2f} "
              f"{br['std_a']:>5.2f}")

    print(f"\n  Team-level results:")
    print(f"    {team_a_name} wins: {result['p_team_a']:.1%}")
    print(f"    {team_b_name} wins: {result['p_team_b']:.1%}")
    print(f"    Draws:         {result['p_draw']:.1%}")
    print(f"    Mean margin:   {result['mean_margin']:+.1f} points")
    print(f"    Margin σ:      {result['std_margin']:.1f} points")

    # Percentiles
    pcts = np.percentile(margin, [5, 25, 50, 75, 95])
    print(f"\n  Margin distribution (A − B):")
    print(f"    5th pct:  {pcts[0]:+.0f}   (worst case for {team_a_name})")
    print(f"    25th pct: {pcts[1]:+.0f}")
    print(f"    Median:   {pcts[2]:+.0f}")
    print(f"    75th pct: {pcts[3]:+.0f}")
    print(f"    95th pct: {pcts[4]:+.0f}   (best case for {team_a_name})")

    if save:
        _plot_match(result, margin, team_a_name, team_b_name)

    return result


def _plot_match(result, margin, team_a_name, team_b_name):
    """Generate match analysis plot: margin histogram + board distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Score margin histogram
    ax = axes[0]
    bins = np.arange(margin.min() - 0.5, margin.max() + 1.5, 1)
    counts, _, patches = ax.hist(margin, bins=bins, edgecolor="white",
                                  alpha=0.8, density=True, color="#5B8DBE")
    # Color negative margin red
    for patch, left_edge in zip(patches, bins):
        if left_edge + 0.5 < 0:
            patch.set_facecolor("#E07B6A")
        elif left_edge + 0.5 == 0:
            patch.set_facecolor("#AAAAAA")

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(margin.mean(), color="#2A6496", linewidth=2, linestyle="-",
               label=f"Mean: {margin.mean():+.1f}")
    ax.set_xlabel("Score Margin (A − B)")
    ax.set_ylabel("Density")
    ax.set_title(f"{team_a_name} vs {team_b_name}\nScore Margin Distribution "
                 f"({N_SIMS:,} sims)")
    ax.legend(fontsize=9)

    # Annotate win probabilities
    ax.text(0.02, 0.95, f"{team_a_name} wins: {result['p_team_a']:.1%}\n"
            f"{team_b_name} wins: {result['p_team_b']:.1%}\n"
            f"Draw: {result['p_draw']:.1%}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # Right: Per-board score distributions (violin-style)
    ax2 = axes[1]
    board_data = []
    for br in result["boards"]:
        for s in br["scores_a"]:
            board_data.append({"Board": f"B{br['board']}: {br['player_a']}\nvs {br['player_b']}",
                               "Score A": s})
    df_board = pd.DataFrame(board_data)

    parts = ax2.violinplot(
        [br["scores_a"] for br in result["boards"]],
        positions=range(len(result["boards"])),
        showmeans=True, showmedians=True
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#5B8DBE")
        pc.set_alpha(0.6)

    ax2.set_xticks(range(len(result["boards"])))
    ax2.set_xticklabels(
        [f"B{br['board']}\n{br['player_a']}\nvs {br['player_b']}"
         for br in result["boards"]],
        fontsize=8
    )
    ax2.axhline(GAMES_PER_BOARD / 2, color="gray", linestyle=":", alpha=0.6)
    ax2.set_ylabel(f"Score for Player A (out of {GAMES_PER_BOARD})")
    ax2.set_title("Per-Board Score Distributions")

    plt.tight_layout()
    safe_name = f"{team_a_name}_vs_{team_b_name}".replace(" ", "_").lower()
    fname = os.path.join(MC_DIR, f"match_{safe_name}.png")
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Upset Probability Analysis
# ═══════════════════════════════════════════════════════════════

def upset_analysis(conn, elo_dict):
    """
    For every historical match, compute:
    - The Elo-predicted favourite
    - MC-simulated upset probability
    - Actual outcome

    This measures how well MC probabilities calibrate against reality.
    """
    matches = pd.read_sql("""
        SELECT m.match_id, m.tournament_round,
               ta.name AS team_a, tb.name AS team_b,
               vm.winner AS actual_winner
        FROM matches m
        JOIN teams ta ON ta.team_id = m.team_a_id
        JOIN teams tb ON tb.team_id = m.team_b_id
        JOIN v_matches vm ON vm.match_id = m.match_id
        ORDER BY m.tournament_round, m.match_id
    """, conn)

    # For each match, get the board lineup
    pairings = pd.read_sql("""
        SELECT sr.match_id,
               pa.nick AS nick_a, pb.nick AS nick_b
        FROM pairings p
        JOIN sub_rounds sr ON sr.sub_round_id = p.sub_round_id
        JOIN players pa ON pa.player_id = p.player_a_id
        JOIN players pb ON pb.player_id = p.player_b_id
    """, conn)

    results = []
    for _, match in matches.iterrows():
        mp = pairings[pairings.match_id == match.match_id]
        if mp.empty:
            continue
        lineup_a = mp.nick_a.tolist()
        lineup_b = mp.nick_b.tolist()

        sim = simulate_match(lineup_a, lineup_b, elo_dict, n_sims=N_SIMS)

        # Determine favourite
        if sim["p_team_a"] > sim["p_team_b"]:
            favourite = "A"
            p_fav = sim["p_team_a"]
        else:
            favourite = "B"
            p_fav = sim["p_team_b"]

        is_upset = (match.actual_winner != favourite and
                    match.actual_winner in ("A", "B"))

        results.append({
            "round":         match.tournament_round,
            "team_a":        match.team_a,
            "team_b":        match.team_b,
            "p_team_a":      round(sim["p_team_a"], 4),
            "p_team_b":      round(sim["p_team_b"], 4),
            "p_draw":        round(sim["p_draw"], 4),
            "favourite":     match.team_a if favourite == "A" else match.team_b,
            "p_favourite":   round(p_fav, 4),
            "actual_winner": match.team_a if match.actual_winner == "A" else
                             (match.team_b if match.actual_winner == "B" else "Draw"),
            "is_upset":      is_upset,
            "mean_margin":   round(sim["mean_margin"], 1),
        })

    df_upset = pd.DataFrame(results)
    df_upset.to_csv(f"{MC_DIR}/upset_analysis.csv", index=False)

    n_upsets = df_upset.is_upset.sum()
    n_total  = len(df_upset)
    print(f"\n  Upset analysis: {n_upsets}/{n_total} matches were upsets "
          f"({n_upsets/n_total:.1%})")
    print(f"\n  {'Round':<6} {'Match':<42} {'P(fav)':>7} {'Actual':<18} {'Upset':>5}")
    print(f"  {'─'*6} {'─'*42} {'─'*7} {'─'*18} {'─'*5}")
    for _, r in df_upset.iterrows():
        mark = "⚡" if r.is_upset else " "
        print(f"  R{int(r['round']):<5} {r.team_a+' vs '+r.team_b:<42} "
              f"{r.p_favourite:>6.1%} {r.actual_winner:<18} {mark:>5}")

    return df_upset


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Calibration Plot
# ═══════════════════════════════════════════════════════════════

def _plot_calibration(df_upset):
    """
    Calibration plot: do MC probabilities match reality?
    Bin predicted probabilities and compare to actual win rate.
    """
    # Use p_team_a as the predicted probability, actual = did A win?
    df = df_upset.copy()
    df["actual_a_win"] = df["actual_winner"] == df["team_a"]
    df["p_bin"] = pd.cut(df["p_team_a"], bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
                         labels=["<35%", "35-45%", "45-55%", "55-65%", ">65%"])

    cal = df.groupby("p_bin", observed=False).agg(
        n=("actual_a_win", "count"),
        actual_rate=("actual_a_win", "mean"),
        mean_pred=("p_team_a", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfect calibration")
    valid = cal[cal.n > 0]
    ax.scatter(valid.mean_pred, valid.actual_rate, s=valid.n * 30,
               color="#5B8DBE", edgecolors="white", linewidths=1.5, zorder=5)
    for _, row in valid.iterrows():
        ax.annotate(f"n={int(row.n)}", (row.mean_pred, row.actual_rate),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("MC Predicted P(Team A wins)")
    ax.set_ylabel("Actual P(Team A wins)")
    ax.set_title("Monte Carlo Calibration Plot\n(bubble size = # matches in bin)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = os.path.join(MC_DIR, "calibration_plot.png")
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Board Sensitivity: "What If" Swap Analysis
# ═══════════════════════════════════════════════════════════════

def board_sensitivity(lineup_a, lineup_b, elo_dict,
                      team_a_name="Team A", team_b_name="Team B"):
    """
    Drop each board player one at a time (replace with Elo=1200 placeholder)
    to measure each player's marginal contribution to team win probability.
    """
    baseline = simulate_match(lineup_a, lineup_b, elo_dict, n_sims=N_SIMS)
    p_base = baseline["p_team_a"]

    print(f"\n  Board Sensitivity — {team_a_name} vs {team_b_name}")
    print(f"  Baseline P({team_a_name} wins) = {p_base:.1%}")
    print(f"\n  {'Swap out':<18} {'Replacement Elo':<16} {'P(win)':>8} {'Δ':>8}")
    print(f"  {'─'*18} {'─'*16} {'─'*8} {'─'*8}")

    deltas = []
    for i, nick in enumerate(lineup_a):
        modified_a = lineup_a.copy()
        modified_elo = elo_dict.copy()
        modified_a[i] = f"__placeholder_{i}__"
        modified_elo[modified_a[i]] = 1200  # average replacement

        result = simulate_match(modified_a, lineup_b, modified_elo, n_sims=N_SIMS)
        delta = result["p_team_a"] - p_base
        deltas.append({"player": nick, "board": i + 1,
                        "p_without": result["p_team_a"], "delta": delta})
        print(f"  {nick:<18} {'Elo 1200':<16} {result['p_team_a']:>7.1%} {delta:>+7.1%}")

    # Also for team B
    print()
    for i, nick in enumerate(lineup_b):
        modified_b = lineup_b.copy()
        modified_elo = elo_dict.copy()
        modified_b[i] = f"__placeholder_{i}__"
        modified_elo[modified_b[i]] = 1200

        result = simulate_match(lineup_a, modified_b, modified_elo, n_sims=N_SIMS)
        delta = result["p_team_a"] - p_base
        deltas.append({"player": nick, "board": i + 1, "side": "B",
                        "p_without": result["p_team_a"], "delta": delta})
        print(f"  {nick:<18} {'Elo 1200':<16} {result['p_team_a']:>7.1%} {delta:>+7.1%}")

    return pd.DataFrame(deltas)


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Full Visualisation: MC Score Distribution Grid
# ═══════════════════════════════════════════════════════════════

def _plot_score_grid(matches_to_plot, elo_dict):
    """
    Plot a grid of match score distributions for selected matchups.
    """
    n = len(matches_to_plot)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (la, lb, name_a, name_b) in enumerate(matches_to_plot):
        ax = axes[idx]
        sim = simulate_match(la, lb, elo_dict, n_sims=N_SIMS)
        margin = sim["total_a"] - sim["total_b"]

        bins = np.arange(margin.min() - 0.5, margin.max() + 1.5, 1)
        _, _, patches = ax.hist(margin, bins=bins, edgecolor="white",
                                alpha=0.8, density=True, color="#5B8DBE")
        for patch, left_edge in zip(patches, bins):
            if left_edge + 0.5 < 0:
                patch.set_facecolor("#E07B6A")

        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(f"{name_a} vs {name_b}\n"
                     f"P(A)={sim['p_team_a']:.0%}  P(B)={sim['p_team_b']:.0%}",
                     fontsize=9)
        ax.set_xlabel("Margin (A−B)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Monte Carlo Score Distributions ({N_SIMS:,} simulations)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fname = os.path.join(MC_DIR, "score_distribution_grid.png")
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Monte Carlo Match & Tournament Simulator")
    print(f"  Simulations per scenario: {N_SIMS:,}")
    print("=" * 60)

    # Load Elo
    elo_df = pd.read_csv(f"{OUT}/elo_ratings_final.csv")
    elo = dict(zip(elo_df.nick, elo_df.elo))
    print(f"Loaded Elo for {len(elo)} players")

    conn = sqlite3.connect(DB)

    # ── Step 1: Featured match deep dives ──
    print("\n" + "═" * 60)
    print("  STEP 1 — Featured Match Simulations")
    print("═" * 60)

    # Poland A vs Czechia A (two strongest teams)
    analyze_match(
        ["wbcbeetle", "wbcbb", "wbcpuholek"],
        ["wbcd",      "wbca",  "wbcc"],
        elo, team_a_name="Poland A", team_b_name="Czechia A"
    )

    # Hungary A vs Russia (different styles)
    analyze_match(
        ["wbczoli",   "wbciron",  "wbcciaran"],
        ["wbcfurla",  "wbcalleb", "wbcbromozel"],
        elo, team_a_name="Hungary A", team_b_name="Russia"
    )

    # ── Step 2: Upset analysis across all historical matches ──
    print("\n" + "═" * 60)
    print("  STEP 2 — Historical Upset Analysis")
    print("═" * 60)
    df_upset = upset_analysis(conn, elo)

    # ── Step 3: Calibration plot ──
    print("\n" + "═" * 60)
    print("  STEP 3 — MC Calibration Check")
    print("═" * 60)
    _plot_calibration(df_upset)

    # ── Step 4: Board sensitivity ──
    print("\n" + "═" * 60)
    print("  STEP 4 — Board Sensitivity (Poland A vs Czechia A)")
    print("═" * 60)
    board_sensitivity(
        ["wbcbeetle", "wbcbb", "wbcpuholek"],
        ["wbcd",      "wbca",  "wbcc"],
        elo, team_a_name="Poland A", team_b_name="Czechia A"
    )

    # ── Step 5: Score distribution grid ──
    print("\n" + "═" * 60)
    print("  STEP 5 — Score Distribution Grid")
    print("═" * 60)

    # Pick matches from DB for grid
    top_matches = pd.read_sql("""
        SELECT m.match_id, ta.name AS team_a, tb.name AS team_b
        FROM matches m
        JOIN teams ta ON ta.team_id = m.team_a_id
        JOIN teams tb ON tb.team_id = m.team_b_id
        WHERE m.tournament_round = 3
        LIMIT 6
    """, conn)

    grid_matchups = []
    for _, match in top_matches.iterrows():
        mp = pd.read_sql(f"""
            SELECT pa.nick AS nick_a, pb.nick AS nick_b
            FROM pairings p
            JOIN sub_rounds sr ON sr.sub_round_id = p.sub_round_id
            JOIN players pa ON pa.player_id = p.player_a_id
            JOIN players pb ON pb.player_id = p.player_b_id
            WHERE sr.match_id = {match.match_id}
        """, conn)
        if not mp.empty:
            grid_matchups.append((
                mp.nick_a.tolist(), mp.nick_b.tolist(),
                match.team_a, match.team_b
            ))

    if grid_matchups:
        _plot_score_grid(grid_matchups, elo)

    # ── Output checklist ──
    print("\n" + "=" * 60)
    print("  OUTPUT CHECKLIST")
    print("=" * 60)
    expected = [
        "monte_carlo/match_poland_a_vs_czechia_a.png",
        "monte_carlo/match_hungary_a_vs_russia.png",
        "monte_carlo/upset_analysis.csv",
        "monte_carlo/calibration_plot.png",
        "monte_carlo/score_distribution_grid.png",
    ]
    for f in expected:
        exists = os.path.isfile(os.path.join(OUT, f))
        mark = "✓" if exists else "✗"
        print(f"  [{mark}] output/{f}")

    conn.close()
    print("\n  Done.")


if __name__ == "__main__":
    run()
