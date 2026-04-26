"""
Phase 2 — Predictive Modeling
==============================
Steps:
  2.1  Elo Rating System (K=32 default / K=16 experienced >60 games)
  2.2  Win Probability Matrix (Elo-based, long-format CSV)
  2.3  Bayesian Head-to-Head (Beta-Binomial, CI + signal vs Elo)
  2.4  Visualizations
        A — Elo progression (top 10 non-outlier players)
        B — Elo vs career efficiency scatter
        C — Win probability heatmap (≥12 career games, non-outliers)
  2.5  Phase 2 Summary

Prerequisite: output/player_career_stats.csv (from phase1_fix.run())
"""
import sqlite3, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import beta as beta_dist
from collections import defaultdict

np.random.seed(42)

DB  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "twbc.db")
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

# ── Elo tuning constants ──────────────────────────────────────
K_DEFAULT        = 32
K_EXPERIENCED    = 16
ELO_INIT         = 1200
CAREER_THRESHOLD = 60   # games threshold for K reduction


# ─────────────────────────────────────────────────────────────
# STEP 2.1 — Elo Rating System
# ─────────────────────────────────────────────────────────────

def _load_prerequisites(conn, out_dir):
    """Load career stats CSV + players info from DB."""
    career_path = os.path.join(out_dir, "player_career_stats.csv")
    if not os.path.isfile(career_path):
        raise FileNotFoundError(
            f"Prerequisite missing: {career_path}\n"
            "Run phase1_fix.run() first to generate player_career_stats.csv."
        )
    career = pd.read_csv(career_path)
    career_games = dict(zip(career.player_nick, career.total_games))
    career_eff   = dict(zip(career.player_nick, career.career_efficiency))
    outlier_nicks = set(career[career.career_efficiency < 0.10].player_nick)
    print(f"Players loaded: {len(career)} | Outliers flagged: {len(outlier_nicks)}")
    return career, career_games, career_eff, outlier_nicks


def _load_pairings(conn):
    """Load pairings in strict chronological order."""
    pairings = pd.read_sql("""
        SELECT
            vp.pairing_id,
            m.tournament_round,
            sr.round_number,
            pa.nick  AS nick_a,
            pb.nick  AS nick_b,
            vp.score_a,
            vp.score_b,
            vp.total_games,
            vp.winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id = vp.sub_round_id
        JOIN matches m     ON m.match_id = sr.match_id
        JOIN players pa    ON pa.player_id = vp.player_a_id
        JOIN players pb    ON pb.player_id = vp.player_b_id
        ORDER BY m.tournament_round, sr.round_number, vp.pairing_id
    """, conn)
    print(f"Total pairings: {len(pairings)}")
    return pairings


def _run_elo(pairings, career_games, all_nicks):
    """
    Run Elo update loop, capturing per-round snapshots.
    Returns:
        elo       : dict nick → final Elo
        snapshots : dict nick → {round_number: elo}
    """
    elo = defaultdict(lambda: ELO_INIT)

    # Pre-seed every known player so snapshots are complete
    for nick in all_nicks:
        _ = elo[nick]   # trigger defaultdict init

    # Snapshots: round 0 = starting Elo for everyone
    snapshots = {nick: {0: ELO_INIT} for nick in all_nicks}

    current_round = 0

    for _, row in pairings.iterrows():
        tr = int(row.tournament_round)

        # When we enter a new tournament round, snapshot previous round's final Elo
        if tr != current_round:
            if current_round > 0:
                for nick in elo:
                    snapshots.setdefault(nick, {})[current_round] = round(elo[nick], 1)
            current_round = tr

        na, nb = row.nick_a, row.nick_b
        ra, rb = elo[na], elo[nb]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea

        if   row.winner == "A":    sa, sb = 1.0, 0.0
        elif row.winner == "B":    sa, sb = 0.0, 1.0
        else:                      sa, sb = 0.5, 0.5

        ka = K_EXPERIENCED if career_games.get(na, 0) > CAREER_THRESHOLD else K_DEFAULT
        kb = K_EXPERIENCED if career_games.get(nb, 0) > CAREER_THRESHOLD else K_DEFAULT

        elo[na] += ka * (sa - ea)
        elo[nb] += kb * (sb - eb)

    # Final snapshot (last tournament round)
    if current_round > 0:
        for nick in elo:
            snapshots.setdefault(nick, {})[current_round] = round(elo[nick], 1)

    return dict(elo), snapshots


def _save_elo_outputs(conn, elo, snapshots, career_games, career_eff, outlier_nicks, out_dir):
    """Build and save elo_ratings_final.csv and elo_progression.csv."""
    players_info = pd.read_sql("""
        SELECT p.nick, p.full_name, t.name AS team
        FROM players p JOIN teams t ON t.team_id = p.team_id
    """, conn)

    # ── Final Elo ratings ──
    df_elo = players_info.copy()
    df_elo["elo"]          = df_elo.nick.map(lambda n: round(elo.get(n, ELO_INIT), 1))
    df_elo["career_games"] = df_elo.nick.map(career_games)
    df_elo["career_eff"]   = df_elo.nick.map(career_eff).round(4)
    df_elo["is_outlier"]   = df_elo.nick.isin(outlier_nicks)
    df_elo = df_elo.sort_values("elo", ascending=False).reset_index(drop=True)
    df_elo.to_csv(f"{out_dir}/elo_ratings_final.csv", index=False)

    print("\nTop 15 players by Elo:")
    print(df_elo[["nick","full_name","team","elo","career_eff","career_games"]].head(15).to_string(index=False))

    # ── Elo progression ──
    rounds_played = sorted({r for snap in snapshots.values() for r in snap if r > 0})
    prog_rows = []
    for nick, snap in snapshots.items():
        row_d = {"nick": nick}
        for r in rounds_played:
            row_d[f"round_{r}_elo"] = round(snap.get(r, ELO_INIT), 1)
        prog_rows.append(row_d)
    df_prog = pd.DataFrame(prog_rows)
    df_prog = df_prog.merge(players_info[["nick","full_name","team"]], on="nick", how="left")
    df_prog.to_csv(f"{out_dir}/elo_progression.csv", index=False)

    print(f"\nSaved: {out_dir}/elo_ratings_final.csv")
    print(f"Saved: {out_dir}/elo_progression.csv")
    return df_elo, df_prog, rounds_played


# ─────────────────────────────────────────────────────────────
# STEP 2.2 — Win Probability Matrix (Elo-based, long format)
# ─────────────────────────────────────────────────────────────

def _win_prob_matrix(conn, elo, out_dir):
    """Compute pairwise Elo win probabilities for all observed pairs."""
    h2h_pairs = pd.read_sql("""
        SELECT DISTINCT pa.nick AS nick_a, pb.nick AS nick_b
        FROM v_pairings vp
        JOIN players pa ON pa.player_id = vp.player_a_id
        JOIN players pb ON pb.player_id = vp.player_b_id
    """, conn)

    records = []
    for _, row in h2h_pairs.iterrows():
        ra = elo.get(row.nick_a, ELO_INIT)
        rb = elo.get(row.nick_b, ELO_INIT)
        p  = 1 / (1 + 10 ** ((rb - ra) / 400))
        records.append({
            "nick_a":      row.nick_a,
            "nick_b":      row.nick_b,
            "p_a_wins_elo": round(p, 4),
        })

    df_winprob = pd.DataFrame(records)
    df_winprob.to_csv(f"{out_dir}/win_prob_matrix_elo.csv", index=False)
    print(f"Saved: {out_dir}/win_prob_matrix_elo.csv  ({len(df_winprob)} pairs)")
    return df_winprob


# ─────────────────────────────────────────────────────────────
# STEP 2.3 — Bayesian Head-to-Head (Beta-Binomial)
# ─────────────────────────────────────────────────────────────

def _bayesian_h2h(conn, elo, out_dir):
    """Beta-Binomial head-to-head with CI and signal vs Elo."""
    ALPHA_PRIOR = 2
    BETA_PRIOR  = 2   # weakly informative, centered at 0.5

    h2h_results = pd.read_sql("""
        SELECT
            pa.nick  AS nick_a,
            pb.nick  AS nick_b,
            COUNT(*) AS games,
            SUM(CASE WHEN vp.winner = 'A'    THEN 1 ELSE 0 END) AS a_wins,
            SUM(CASE WHEN vp.winner = 'B'    THEN 1 ELSE 0 END) AS b_wins,
            SUM(CASE WHEN vp.winner = 'Draw' THEN 1 ELSE 0 END) AS draws
        FROM v_pairings vp
        JOIN players pa ON pa.player_id = vp.player_a_id
        JOIN players pb ON pb.player_id = vp.player_b_id
        GROUP BY vp.player_a_id, vp.player_b_id
        HAVING COUNT(*) >= 1
    """, conn)

    bayes_rows = []
    for _, row in h2h_results.iterrows():
        alpha = ALPHA_PRIOR + row.a_wins
        beta  = BETA_PRIOR  + row.b_wins
        p_a   = alpha / (alpha + beta)
        ci_lo = beta_dist.ppf(0.025, alpha, beta)
        ci_hi = beta_dist.ppf(0.975, alpha, beta)
        elo_p = 1 / (1 + 10 ** ((elo.get(row.nick_b, ELO_INIT) -
                                  elo.get(row.nick_a, ELO_INIT)) / 400))
        bayes_rows.append({
            "nick_a":      row.nick_a,
            "nick_b":      row.nick_b,
            "games":       int(row.games),
            "a_wins":      int(row.a_wins),
            "b_wins":      int(row.b_wins),
            "draws":       int(row.draws),
            "p_a_wins":    round(p_a, 4),
            "ci_lower":    round(ci_lo, 4),
            "ci_upper":    round(ci_hi, 4),
            "uncertainty": round(ci_hi - ci_lo, 4),
            "elo_p_a":     round(elo_p, 4),
        })

    df_bayes = pd.DataFrame(bayes_rows)
    df_bayes["signal"] = (df_bayes.p_a_wins - df_bayes.elo_p_a).round(4)
    df_bayes.to_csv(f"{out_dir}/bayesian_head_to_head.csv", index=False)
    print(f"Saved: {out_dir}/bayesian_head_to_head.csv  ({len(df_bayes)} pairs)")

    # Notable divergences
    divergences = df_bayes[
        (df_bayes.games >= 2) & (df_bayes.signal.abs() >= 0.15)
    ].sort_values("signal", ascending=False)

    print(f"\nNotable Elo vs Bayesian divergences ({len(divergences)}):")
    if len(divergences):
        print(divergences[["nick_a","nick_b","games","p_a_wins",
                            "elo_p_a","signal","uncertainty"]].to_string(index=False))
    return df_bayes


# ─────────────────────────────────────────────────────────────
# STEP 2.4 — Visualizations
# ─────────────────────────────────────────────────────────────

def _plot_elo_progression(df_elo, df_prog, rounds_played, outlier_nicks, out_dir):
    """Plot A — Elo progression for top 10 non-outlier players."""
    top10_nicks = df_elo[~df_elo.is_outlier].head(10).nick.tolist()
    round_cols  = [f"round_{r}_elo" for r in rounds_played]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps["tab10"]

    for i, nick in enumerate(top10_nicks):
        row = df_prog[df_prog.nick == nick]
        if row.empty:
            continue
        vals = [ELO_INIT] + [float(row[c].values[0]) for c in round_cols]
        xs   = list(range(len(vals)))
        color = cmap(i)
        ax.plot(xs, vals, "o-", color=color, lw=2, label=nick)
        ax.annotate(nick, xy=(xs[-1], vals[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    ax.set_xticks(range(len(round_cols) + 1))
    ax.set_xticklabels(["Start"] + [f"Round {r}" for r in rounds_played])
    ax.axhline(ELO_INIT, color="gray", lw=1, linestyle=":", alpha=0.6, label="Initial (1200)")
    ax.set_ylabel("Elo Rating")
    ax.set_title("Elo progression — top 10 players")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plot_elo_progression.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved: plot_elo_progression.png")


def _plot_elo_vs_efficiency(df_elo, out_dir):
    """Plot B — Elo vs career efficiency scatter (bubble size = games)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    teams_list = df_elo.team.unique()
    # Use tab20 but fall back gracefully if fewer teams
    palette = dict(zip(teams_list, plt.cm.tab20.colors[:len(teams_list)]))

    for team, grp in df_elo.groupby("team"):
        non_out = grp[~grp.is_outlier]
        out_pts = grp[grp.is_outlier]
        color = palette[team]
        if len(non_out):
            ax.scatter(non_out.career_eff, non_out.elo,
                       s=non_out.career_games * 1.8,
                       color=color, alpha=0.75, label=team,
                       marker="o", edgecolors="white", linewidths=0.5, zorder=3)
        if len(out_pts):
            ax.scatter(out_pts.career_eff, out_pts.elo,
                       s=out_pts.career_games * 1.8,
                       color=color, alpha=0.75,
                       marker="x", linewidths=1.5, zorder=3)

    # Annotate top 10 and bottom 5
    for _, row in df_elo.head(10).iterrows():
        ax.annotate(row.nick, (row.career_eff, row.elo),
                    fontsize=7.5, xytext=(4, 4),
                    textcoords="offset points", alpha=0.9)
    for _, row in df_elo.tail(5).iterrows():
        ax.annotate(row.nick, (row.career_eff, row.elo),
                    fontsize=7.5, xytext=(4, -10),
                    textcoords="offset points", alpha=0.7, color="gray")

    ax.axhline(ELO_INIT, color="gray", lw=1, linestyle=":", alpha=0.5)
    ax.axvline(0.5,      color="gray", lw=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Career Efficiency")
    ax.set_ylabel("Final Elo Rating")
    ax.set_title("Final Elo vs Career Efficiency\n(bubble size = games played, × = outlier player)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Team", title_fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scatter_elo_vs_efficiency.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved: scatter_elo_vs_efficiency.png")


def _plot_winprob_heatmap(df_elo, df_winprob, out_dir):
    """Plot C — Win probability heatmap (players ≥12 games, non-outliers)."""
    eligible = df_elo[
        (df_elo.career_games >= 12) & (~df_elo.is_outlier)
    ].nick.tolist()

    n = len(eligible)
    if n < 2:
        print("  ⚠ Not enough eligible players for heatmap — skipped.")
        return

    # Build square matrix defaulting to 0.5
    matrix = pd.DataFrame(0.5, index=eligible, columns=eligible)

    for _, row in df_winprob.iterrows():
        if row.nick_a in matrix.index and row.nick_b in matrix.columns:
            matrix.loc[row.nick_a, row.nick_b] = row.p_a_wins_elo
            matrix.loc[row.nick_b, row.nick_a] = round(1 - row.p_a_wins_elo, 4)

    np.fill_diagonal(matrix.values, np.nan)   # diagonal = undefined

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.5)))
    sns.heatmap(
        matrix, ax=ax,
        cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
        annot=(n <= 25),
        fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3,
        cbar_kws={"label": "P(row player beats column player)"}
    )
    ax.set_title(
        "Elo-based win probability matrix\n(row = player A, col = player B)", fontsize=12
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_winprob_elo.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: heatmap_winprob_elo.png  ({n}×{n} players)")


# ─────────────────────────────────────────────────────────────
# STEP 2.5 — Phase 2 Summary
# ─────────────────────────────────────────────────────────────

def _print_summary(df_elo, df_bayes):
    print("""
╔══════════════════════════════════════════════════════════════╗
║              PHASE 2 SUMMARY                                ║
╠══════════════════════════════════════════════════════════════╣""")

    top5 = df_elo[~df_elo.is_outlier].head(5)
    print("║  Top 5 by final Elo:                                        ║")
    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        line = f"║    {rank}. {row.nick:<16} {row.elo:>7.1f}  ({row.team})"
        print(f"{line:<63}║")

    high_signal = df_bayes[df_bayes.signal >= 0.15].sort_values("signal", ascending=False)
    print("║                                                             ║")
    print("║  Pairs where actual record >> Elo prediction (signal≥0.15):║")
    if len(high_signal):
        for _, row in high_signal.head(3).iterrows():
            line = f"║    {row.nick_a} vs {row.nick_b}: signal={row.signal:+.2f} ({row.games}g)"
            print(f"{line:<63}║")
    else:
        print("║    (none)                                                   ║")

    print("""╠══════════════════════════════════════════════════════════════╣
║  Next: Phase 3 — Adjusted Efficiency + SOS + Validation     ║
╚══════════════════════════════════════════════════════════════╝""")

    # Output checklist
    print("""
OUTPUT CHECKLIST:
  [ ] output/elo_ratings_final.csv
  [ ] output/elo_progression.csv
  [ ] output/win_prob_matrix_elo.csv
  [ ] output/bayesian_head_to_head.csv
  [ ] output/plot_elo_progression.png
  [ ] output/scatter_elo_vs_efficiency.png
  [ ] output/heatmap_winprob_elo.png
""")


# ─────────────────────────────────────────────────────────────
# PUBLIC COMPATIBILITY API  (imported by phase3_validate)
# ─────────────────────────────────────────────────────────────

def compute_elo(pairings_df, career_games, max_round=None):
    """
    Public wrapper kept for backward-compatibility with phase3_validate.

    Parameters
    ----------
    pairings_df   : DataFrame with columns tournament_round, nick_a, nick_b, winner
    career_games  : dict nick → total career games
    max_round     : int or None — if set, only pairings up to this round are used

    Returns
    -------
    elo           : dict  nick → final Elo float
    snapshots     : dict  nick → {round_number: elo}
    """
    if max_round is not None:
        pairings_df = pairings_df[pairings_df["tournament_round"] <= max_round].copy()

    all_nicks = set(pairings_df.nick_a) | set(pairings_df.nick_b)
    return _run_elo(pairings_df, career_games, all_nicks)


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run():
    conn = sqlite3.connect(DB)

    # Prerequisites
    career, career_games, career_eff, outlier_nicks = _load_prerequisites(conn, OUT)

    # ── 2.1 Elo ──
    print("\n" + "─" * 55)
    print("  STEP 2.1 — Elo Rating System")
    print("─" * 55)
    pairings = _load_pairings(conn)

    # All known player nicks (union of career + pairings)
    all_nicks = set(career.player_nick) | set(pairings.nick_a) | set(pairings.nick_b)

    elo, snapshots = _run_elo(pairings, career_games, all_nicks)
    df_elo, df_prog, rounds_played = _save_elo_outputs(
        conn, elo, snapshots, career_games, career_eff, outlier_nicks, OUT
    )

    # ── 2.2 Win probability matrix ──
    print("\n" + "─" * 55)
    print("  STEP 2.2 — Win Probability Matrix")
    print("─" * 55)
    df_winprob = _win_prob_matrix(conn, elo, OUT)

    # ── 2.3 Bayesian head-to-head ──
    print("\n" + "─" * 55)
    print("  STEP 2.3 — Bayesian Head-to-Head")
    print("─" * 55)
    df_bayes = _bayesian_h2h(conn, elo, OUT)

    # ── 2.4 Visualizations ──
    print("\n" + "─" * 55)
    print("  STEP 2.4 — Visualizations")
    print("─" * 55)
    _plot_elo_progression(df_elo, df_prog, rounds_played, outlier_nicks, OUT)
    _plot_elo_vs_efficiency(df_elo, OUT)
    _plot_winprob_heatmap(df_elo, df_winprob, OUT)

    # ── 2.5 Summary ──
    print("\n" + "─" * 55)
    print("  STEP 2.5 — Phase 2 Summary")
    print("─" * 55)
    _print_summary(df_elo, df_bayes)

    conn.close()
    return elo, snapshots, career_games, pairings


if __name__ == "__main__":
    run()
