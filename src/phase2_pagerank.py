"""
Phase 2C — Bradley-Terry / Eigenvector Rating
===============================================
Uses maximum-likelihood estimation to compute player strength
parameters from pairwise comparison data.

Bradley-Terry model:
  P(i beats j) = γ_i / (γ_i + γ_j)

Solved via Zermelo's iterative algorithm:
  γ_i ← w_i / Σ_j [n_ij / (γ_i + γ_j)]

  w_i   = total wins of player i (draws count as 0.5)
  n_ij  = total games between i and j

Advantages over Elo:
  - Path-independent: result does not depend on match ordering
  - Transitive: beating a strong player is worth more
  - Global MLE: uses all data simultaneously
  - Well-defined for sparse data (unlike PageRank)

Prerequisites:
  output/elo_ratings_final.csv
  data/processed/twbc.db
"""
import sqlite3, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import DB, data, plot_r, ensure_dirs

np.random.seed(42)
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

MAX_ITER = 500
TOL      = 1e-8


# ═══════════════════════════════════════════════════════════════
# Core: Bradley-Terry MLE via Zermelo's algorithm
# ═══════════════════════════════════════════════════════════════

def build_matrices(pairings: pd.DataFrame, nick_to_idx: dict):
    """
    Build wins vector and games matrix from pairings.

    Returns
    -------
    wins  : array (n,)   — total wins per player (draws=0.5)
    games : array (n, n) — total games between each pair
    """
    n = len(nick_to_idx)
    wins  = np.zeros(n)
    games = np.zeros((n, n))

    for _, row in pairings.iterrows():
        i = nick_to_idx.get(row.nick_a)
        j = nick_to_idx.get(row.nick_b)
        if i is None or j is None:
            continue
        games[i, j] += 1.0
        games[j, i] += 1.0   # symmetric: n_ij = n_ji
        if row.winner == "A":
            wins[i] += 1.0
        elif row.winner == "B":
            wins[j] += 1.0
        else:  # Draw
            wins[i] += 0.5
            wins[j] += 0.5

    return wins, games


def bradley_terry(wins: np.ndarray, games: np.ndarray,
                  max_iter: int = MAX_ITER, tol: float = TOL) -> np.ndarray:
    """
    Zermelo's iterative algorithm for Bradley-Terry MLE.

    γ_i ← w_i / Σ_j [n_ij / (γ_i + γ_j)]

    Converges to the unique MLE (up to scale) when the comparison
    graph is connected.
    """
    n = len(wins)
    gamma = np.ones(n)  # initial strengths

    for iteration in range(max_iter):
        gamma_old = gamma.copy()

        for i in range(n):
            if wins[i] == 0:
                gamma[i] = 1e-6   # player with zero wins → near-zero strength
                continue
            denom = 0.0
            for j in range(n):
                if games[i, j] > 0:
                    denom += games[i, j] / (gamma[i] + gamma[j])
            if denom > 0:
                gamma[i] = wins[i] / denom

        # Normalize so they sum to n (keeps scale stable)
        gamma = gamma / gamma.sum() * n

        # Check convergence
        change = np.max(np.abs(gamma - gamma_old))
        if change < tol:
            print(f"  Converged in {iteration + 1} iterations (max change: {change:.2e})")
            break
    else:
        print(f"  Warning: did not converge in {max_iter} iterations (change: {change:.2e})")

    return gamma


def bt_win_probability(gamma_i: float, gamma_j: float) -> float:
    """P(i beats j) = γ_i / (γ_i + γ_j)."""
    total = gamma_i + gamma_j
    if total < 1e-12:
        return 0.5
    return gamma_i / total


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load_pairings(conn, max_round=None):
    sql = """
        SELECT m.tournament_round, pa.nick AS nick_a, pb.nick AS nick_b,
               vp.total_games, vp.winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id = vp.sub_round_id
        JOIN matches m     ON m.match_id = sr.match_id
        JOIN players pa    ON pa.player_id = vp.player_a_id
        JOIN players pb    ON pb.player_id = vp.player_b_id
        ORDER BY m.tournament_round, sr.round_number, vp.pairing_id
    """
    df = pd.read_sql(sql, conn)
    if max_round is not None:
        df = df[df.tournament_round <= max_round]
    return df


# ═══════════════════════════════════════════════════════════════
# Validation on Round 3
# ═══════════════════════════════════════════════════════════════

def _validate_round3(conn, gamma_r12, nick_to_idx):
    """Predict Round 3 matches using BT trained on R1+R2."""
    r3 = pd.read_sql("""
        SELECT m.match_id, vm.team_a, vm.team_b, vm.winner AS actual,
               pa.nick AS nick_a, pb.nick AS nick_b, vp.total_games
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id = vp.sub_round_id
        JOIN matches m     ON m.match_id = sr.match_id
        JOIN v_matches vm  ON vm.match_id = m.match_id
        JOIN players pa    ON pa.player_id = vp.player_a_id
        JOIN players pb    ON pb.player_id = vp.player_b_id
        WHERE m.tournament_round = 3
    """, conn)

    rows = []
    for mid, grp in r3.groupby("match_id"):
        score_a = score_b = 0.0
        actual = grp.actual.iloc[0]
        for _, p in grp.iterrows():
            ia = nick_to_idx.get(p.nick_a)
            ib = nick_to_idx.get(p.nick_b)
            if ia is not None and ib is not None:
                prob = bt_win_probability(gamma_r12[ia], gamma_r12[ib])
            else:
                prob = 0.5
            score_a += prob * p.total_games
            score_b += (1 - prob) * p.total_games

        pred = "A" if score_a > score_b else "B"
        rows.append({
            "match_id": mid,
            "team_a": grp.team_a.iloc[0],
            "team_b": grp.team_b.iloc[0],
            "bt_pred": pred, "actual": actual,
            "bt_ok": pred == actual,
            "bt_score_a": round(score_a, 2),
            "bt_score_b": round(score_b, 2),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def _plot_comparison(comp_df, top_n=20):
    """Rank comparison: BT vs Elo (vs TrueSkill if available)."""
    df = comp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df.elo_rank, df.bt_rank,
                    c=df.rank_delta, cmap="RdBu", vmin=-15, vmax=15,
                    s=60, edgecolors="white", lw=0.5, alpha=0.85, zorder=3)
    lim = max(df.elo_rank.max(), df.bt_rank.max()) + 1
    ax.plot([1, lim], [1, lim], "--", color="gray", alpha=0.5, lw=1)

    for _, r in df[df.rank_delta.abs() >= 3].head(8).iterrows():
        ax.annotate(f"{r.nick} ({r.rank_delta:+.0f})",
                    (r.elo_rank, r.bt_rank), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5)

    plt.colorbar(sc, ax=ax, label="BT rank − Elo rank (blue = BT ranks higher)")
    ax.set_xlabel("Elo Rank")
    ax.set_ylabel("Bradley-Terry Rank")
    ax.set_title("Bradley-Terry vs Elo Ranking  (MLE vs sequential)")
    plt.tight_layout()
    path = plot_r("plot_bt_vs_elo.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Phase 2C — Bradley-Terry Rating (MLE)")
    print("=" * 60)

    conn = sqlite3.connect(DB)

    # Load all pairings
    pairings_all = _load_pairings(conn)
    all_nicks = sorted(set(pairings_all.nick_a) | set(pairings_all.nick_b))
    nick_to_idx = {n: i for i, n in enumerate(all_nicks)}
    idx_to_nick = {i: n for n, i in nick_to_idx.items()}
    n = len(all_nicks)
    print(f"  Players: {n}  |  Pairings: {len(pairings_all)}")

    # ── Full Bradley-Terry ──
    print("\n── Full Bradley-Terry (all rounds) ──")
    wins_full, games_full = build_matrices(pairings_all, nick_to_idx)
    gamma_full = bradley_terry(wins_full, games_full)

    # Build ratings DataFrame
    players_info = pd.read_sql("""
        SELECT p.nick, p.full_name, t.name AS team
        FROM players p JOIN teams t ON t.team_id = p.team_id
    """, conn)

    rows = [{"nick": idx_to_nick[i], "bt_strength": round(gamma_full[i], 6)}
            for i in range(n)]
    bt_df = (pd.DataFrame(rows)
             .sort_values("bt_strength", ascending=False)
             .reset_index(drop=True))
    bt_df["bt_rank"] = bt_df.index + 1
    bt_df = bt_df.merge(players_info, on="nick", how="left")

    print("\n  Top 15 by Bradley-Terry strength:")
    print(bt_df[["bt_rank", "nick", "full_name", "team", "bt_strength"]].head(15).to_string(index=False))
    bt_df.to_csv(data("bradleyterry_ratings.csv"), index=False)
    print(f"\n  Saved: data/bradleyterry_ratings.csv")

    # ── R1+R2 for validation ──
    print("\n── Bradley-Terry trained on R1+R2 ──")
    pairings_r12 = _load_pairings(conn, max_round=2)
    wins_r12, games_r12 = build_matrices(pairings_r12, nick_to_idx)
    gamma_r12 = bradley_terry(wins_r12, games_r12)

    # ── Validation ──
    print("\n── Round 3 Validation ──")
    val_df = _validate_round3(conn, gamma_r12, nick_to_idx)
    bt_acc = val_df.bt_ok.mean()

    print(f"\n  {'Match':<45} {'BT':>4} {'Actual':>8}")
    print(f"  {'─'*60}")
    for _, r in val_df.iterrows():
        mark = "✓" if r.bt_ok else "✗"
        print(f"  {r.team_a+' vs '+r.team_b:<45} {mark:>4} {r.actual:>8}")
    print(f"\n  Bradley-Terry accuracy: {bt_acc:.1%}  ({val_df.bt_ok.sum()}/{len(val_df)})")
    val_df.to_csv(data("bradleyterry_validation_r3.csv"), index=False)

    # ── Compare with Elo ──
    print("\n── Ranking Comparison ──")
    elo_df = pd.read_csv(data("elo_ratings_final.csv"))
    elo_df["elo_rank"] = elo_df["elo"].rank(ascending=False, method="min").astype(int)

    comp = bt_df[["nick", "bt_rank", "bt_strength", "team"]].merge(
        elo_df[["nick", "elo", "elo_rank", "career_games"]], on="nick", how="left"
    )
    comp["rank_delta"] = comp.elo_rank - comp.bt_rank
    comp = comp.sort_values("bt_rank")
    print(comp.head(15).to_string(index=False))

    _plot_comparison(comp)

    # ── Summary ──
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PHASE 2C — BRADLEY-TERRY SUMMARY                           ║
╠══════════════════════════════════════════════════════════════╣
║  Round 3 validation accuracy: {bt_acc:.1%}  ({val_df.bt_ok.sum()}/{len(val_df)})                  ║
║                                                              ║
║  Bradley-Terry vs Elo:                                       ║
║  - BT is path-independent (global MLE)                      ║
║  - BT uses all data simultaneously                          ║
║  - BT produces win probabilities directly                   ║
╚══════════════════════════════════════════════════════════════╝""")

    conn.close()
    print("\n  Done.")
    return gamma_full, bt_df, val_df


if __name__ == "__main__":
    run()
