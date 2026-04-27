"""
Phase 2B — TrueSkill (scratch implementation, scipy only)
==========================================================
Implements the TrueSkill factor-graph update equations using
scipy.stats.norm — no external 'trueskill' package required.

Key parameters
--------------
MU       = 25.0          initial mean skill
SIGMA    = 25/3 ≈ 8.33   initial uncertainty
BETA     = 25/6 ≈ 4.17   per-game performance noise
TAU      = 25/300        dynamic factor (σ drift per round)
DRAW_PROB               estimated from data

Outputs
-------
output/trueskill_ratings.csv
output/trueskill_vs_elo.csv
output/plot_trueskill_ratings.png
output/plot_ts_vs_elo.png
"""
import sqlite3, os, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from paths import DB, data, plot_r, ensure_dirs

np.random.seed(42)
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

# ── TrueSkill constants ───────────────────────────────────────
MU    = 25.0
SIGMA = MU / 3          # 8.333
BETA  = MU / 6          # 4.167
TAU   = MU / 300        # 0.083  — small drift between rounds


# ═══════════════════════════════════════════════════════════════
# Core math
# ═══════════════════════════════════════════════════════════════

def _v_win(t):
    """Additive correction factor (win case)."""
    denom = norm.cdf(t)
    return norm.pdf(t) / denom if denom > 1e-12 else -t

def _w_win(t):
    v = _v_win(t)
    return v * (v + t)

def _v_draw(t, eps):
    """Additive correction factor (draw case)."""
    a = norm.cdf(eps - t) - norm.cdf(-eps - t)
    if a < 1e-12:
        return 0.0
    return (norm.pdf(-eps - t) - norm.pdf(eps - t)) / a

def _w_draw(t, eps):
    a = norm.cdf(eps - t) - norm.cdf(-eps - t)
    if a < 1e-12:
        return 1.0
    num = (eps - t) * norm.pdf(eps - t) + (eps + t) * norm.pdf(-eps - t)
    return num / a + (norm.cdf(eps - t) - norm.cdf(-eps - t)) ** 2 / a**2

def _draw_margin(draw_prob, n_players=2):
    """Convert draw_probability → ε (draw margin)."""
    return norm.ppf((draw_prob + 1) / 2) * math.sqrt(n_players) * BETA

def win_probability(mu_a, sigma_a, mu_b, sigma_b):
    """P(a beats b) — analytical, no simulation needed."""
    c = math.sqrt(sigma_a**2 + sigma_b**2 + 2 * BETA**2)
    return norm.cdf((mu_a - mu_b) / c)

def conservative(mu, sigma):
    return mu - 3 * sigma


class Rating:
    __slots__ = ("mu", "sigma")
    def __init__(self, mu=MU, sigma=SIGMA):
        self.mu    = mu
        self.sigma = sigma
    def __repr__(self):
        return f"Rating(μ={self.mu:.3f}, σ={self.sigma:.3f})"


def update_win(winner: Rating, loser: Rating, eps: float) -> tuple:
    """Return updated (winner, loser) ratings after a win result."""
    c2 = winner.sigma**2 + loser.sigma**2 + 2 * BETA**2
    c  = math.sqrt(c2)
    t  = (winner.mu - loser.mu) / c

    v = _v_win(t - eps / c)
    w = _w_win(t - eps / c)

    new_winner = Rating(
        mu    = winner.mu    + (winner.sigma**2 / c) * v,
        sigma = math.sqrt(winner.sigma**2 * (1 - (winner.sigma**2 / c2) * w))
    )
    new_loser = Rating(
        mu    = loser.mu    - (loser.sigma**2 / c) * v,
        sigma = math.sqrt(loser.sigma**2  * (1 - (loser.sigma**2  / c2) * w))
    )
    return new_winner, new_loser


def update_draw(r_a: Rating, r_b: Rating, eps: float) -> tuple:
    """Return updated (r_a, r_b) ratings after a draw."""
    c2 = r_a.sigma**2 + r_b.sigma**2 + 2 * BETA**2
    c  = math.sqrt(c2)
    t  = (r_a.mu - r_b.mu) / c

    v = _v_draw(t, eps / c)
    w = _w_draw(t, eps / c)

    def _upd(r, sign):
        new_mu    = r.mu    + sign * (r.sigma**2 / c) * v
        new_sigma = math.sqrt(r.sigma**2 * (1 - (r.sigma**2 / c2) * w))
        return Rating(new_mu, new_sigma)

    return _upd(r_a, +1), _upd(r_b, -1)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load_pairings(conn):
    return pd.read_sql("""
        SELECT vp.pairing_id, m.tournament_round, sr.round_number,
               pa.nick AS nick_a, pb.nick AS nick_b,
               vp.score_a, vp.score_b, vp.total_games, vp.winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id = vp.sub_round_id
        JOIN matches m     ON m.match_id = sr.match_id
        JOIN players pa    ON pa.player_id = vp.player_a_id
        JOIN players pb    ON pb.player_id = vp.player_b_id
        ORDER BY m.tournament_round, sr.round_number, vp.pairing_id
    """, conn)


# ═══════════════════════════════════════════════════════════════
# TrueSkill update loop
# ═══════════════════════════════════════════════════════════════

def _run_ts(pairings: pd.DataFrame, draw_prob: float,
            max_round: int = None) -> dict:
    """
    Sequential TrueSkill update, chronological.
    Returns dict nick → Rating.
    """
    all_nicks = set(pairings.nick_a) | set(pairings.nick_b)
    ratings   = {n: Rating() for n in all_nicks}
    eps       = _draw_margin(draw_prob)

    if max_round is not None:
        pairings = pairings[pairings.tournament_round <= max_round]

    for _, row in pairings.iterrows():
        na, nb = row.nick_a, row.nick_b
        # TAU drift: inflate σ slightly each pairing (simulates inactivity noise)
        for n in (na, nb):
            r = ratings[n]
            ratings[n] = Rating(r.mu, math.sqrt(r.sigma**2 + TAU**2))

        ra, rb = ratings[na], ratings[nb]

        if row.winner == "A":
            ratings[na], ratings[nb] = update_win(ra, rb, eps)
        elif row.winner == "B":
            ratings[nb], ratings[na] = update_win(rb, ra, eps)
        else:
            ratings[na], ratings[nb] = update_draw(ra, rb, eps)

    return ratings


# ═══════════════════════════════════════════════════════════════
# Output builders
# ═══════════════════════════════════════════════════════════════

def _build_df(ratings, conn, career):
    players = pd.read_sql("""
        SELECT p.nick, p.full_name, t.name AS team
        FROM players p JOIN teams t ON t.team_id = p.team_id
    """, conn)

    rows = [{"nick": n, "mu": round(r.mu, 4),
             "sigma": round(r.sigma, 4),
             "conservative": round(conservative(r.mu, r.sigma), 4)}
            for n, r in ratings.items()]

    df = (pd.DataFrame(rows)
          .merge(players, on="nick", how="left")
          .merge(career[["player_nick","total_games","career_efficiency"]]
                       .rename(columns={"player_nick":"nick"}),
                 on="nick", how="left"))
    df = df.sort_values("conservative", ascending=False).reset_index(drop=True)
    df["ts_rank"] = df.index + 1
    return df


def _build_vs_elo(ts_df, elo_df):
    elo = elo_df.copy()
    elo["elo_rank"] = elo["elo"].rank(ascending=False, method="min").astype(int)
    m = ts_df[["nick","mu","sigma","conservative","ts_rank"]].merge(
        elo[["nick","elo","elo_rank","career_games","is_outlier"]], on="nick")
    m["rank_delta"] = (m.elo_rank - m.ts_rank)
    return m.sort_values("ts_rank").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Validation: TrueSkill vs Elo on Round 3
# ═══════════════════════════════════════════════════════════════

def _validate(conn, ratings_r12, elo_df):
    from phase2_elo import compute_elo
    pairings_all = pd.read_sql("""
        SELECT vp.pairing_id, m.tournament_round, sr.round_number,
               pa.nick AS nick_a, pb.nick AS nick_b,
               vp.score_a, vp.score_b, vp.total_games, vp.winner
        FROM v_pairings vp
        JOIN sub_rounds sr ON sr.sub_round_id = vp.sub_round_id
        JOIN matches m     ON m.match_id = sr.match_id
        JOIN players pa    ON pa.player_id = vp.player_a_id
        JOIN players pb    ON pb.player_id = vp.player_b_id
        ORDER BY m.tournament_round, sr.round_number, vp.pairing_id
    """, conn)
    career_games = dict(zip(
        pd.read_csv(data("player_career_stats.csv")).player_nick,
        pd.read_csv(data("player_career_stats.csv")).total_games
    ))
    elo_r12, _ = compute_elo(pairings_all, career_games, max_round=2)

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
        ts_a = ts_b = elo_a = elo_b = 0.0
        actual = grp.actual.iloc[0]
        for _, p in grp.iterrows():
            na, nb = p.nick_a, p.nick_b
            ng = p.total_games
            ra = ratings_r12.get(na, Rating())
            rb = ratings_r12.get(nb, Rating())
            p_ts = win_probability(ra.mu, ra.sigma, rb.mu, rb.sigma)
            ts_a += p_ts * ng;  ts_b += (1 - p_ts) * ng

            ea = elo_r12.get(na, 1200); eb = elo_r12.get(nb, 1200)
            p_elo = 1 / (1 + 10**((eb - ea) / 400))
            elo_a += p_elo * ng;  elo_b += (1 - p_elo) * ng

        ts_pred  = "A" if ts_a  > ts_b  else "B"
        elo_pred = "A" if elo_a > elo_b else "B"
        rows.append({"match_id": mid,
                     "team_a": grp.team_a.iloc[0], "team_b": grp.team_b.iloc[0],
                     "ts_pred": ts_pred, "elo_pred": elo_pred, "actual": actual,
                     "ts_ok": ts_pred == actual, "elo_ok": elo_pred == actual})

    df = pd.DataFrame(rows)
    ts_acc  = df.ts_ok.mean()
    elo_acc = df.elo_ok.mean()

    print(f"\n  {'Match':<45} {'TS':>4} {'Elo':>4} {'Actual':>8}")
    print(f"  {'─'*65}")
    for _, r in df.iterrows():
        print(f"  {r.team_a+' vs '+r.team_b:<45}"
              f" {'✓' if r.ts_ok else '✗':>4}"
              f" {'✓' if r.elo_ok else '✗':>4}"
              f" {r.actual:>8}")
    print(f"\n  TrueSkill accuracy: {ts_acc:.1%}  ({df.ts_ok.sum()}/{len(df)})")
    print(f"  Elo      accuracy: {elo_acc:.1%}  ({df.elo_ok.sum()}/{len(df)})")
    df.to_csv(data("trueskill_validation_r3.csv"), index=False)
    return ts_acc, elo_acc


# ═══════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════

def _plot_ratings(ts_df, top_n=20):
    df = ts_df.head(top_n)
    y  = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.42)))
    ax.barh(y, df["conservative"], color="#4C72B0", alpha=0.75, height=0.6,
            label="Conservative (μ−3σ)")
    ax.errorbar(df["mu"], y, xerr=2*df["sigma"], fmt="none",
                color="#C44E52", lw=1.5, capsize=4, label="μ ± 2σ")
    ax.scatter(df["mu"], y, color="#C44E52", zorder=5, s=35)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.nick}  ({r.team})" for _, r in df.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("TrueSkill Rating")
    ax.set_title(f"TrueSkill — Top {top_n} Players (bar=conservative, error=95% CI)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_r("plot_trueskill_ratings.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/ratings/plot_trueskill_ratings.png")


def _plot_vs_elo(vs_df):
    df = vs_df[~vs_df.is_outlier]
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df.elo_rank, df.ts_rank,
                    c=df.rank_delta, cmap="RdBu", vmin=-10, vmax=10,
                    s=df.career_games*1.5+20, edgecolors="white", lw=0.5,
                    alpha=0.85, zorder=3)
    lim = max(df.elo_rank.max(), df.ts_rank.max()) + 1
    ax.plot([1, lim], [1, lim], "--", color="gray", alpha=0.5, lw=1)
    for _, r in df[df.rank_delta.abs() >= 3].head(8).iterrows():
        ax.annotate(f"{r.nick} ({r.rank_delta:+d})",
                    (r.elo_rank, r.ts_rank), xytext=(5,3),
                    textcoords="offset points", fontsize=7.5)
    plt.colorbar(sc, ax=ax, label="TrueSkill rank − Elo rank")
    ax.set_xlabel("Elo Rank"); ax.set_ylabel("TrueSkill Rank")
    ax.set_title("TrueSkill vs Elo Ranking  (bubble=career games)")
    plt.tight_layout()
    plt.savefig(plot_r("plot_ts_vs_elo.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/ratings/plot_ts_vs_elo.png")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Phase 2B — TrueSkill (scipy, no external package)")
    print("=" * 60)

    conn   = sqlite3.connect(DB)
    career = pd.read_csv(data("player_career_stats.csv"))
    elo_df = pd.read_csv(data("elo_ratings_final.csv"))

    pairings  = _load_pairings(conn)
    draw_prob = (pairings.winner == "Draw").sum() / len(pairings)
    draw_prob = max(draw_prob, 0.01)
    print(f"  Empirical draw rate: {draw_prob:.2%}  |  ε = {_draw_margin(draw_prob):.4f}")
    print(f"  TrueSkill params: μ={MU}, σ={SIGMA:.3f}, β={BETA:.3f}, τ={TAU:.4f}")

    # Full training
    print("\n── Full TrueSkill (all rounds) ──")
    ratings_full = _run_ts(pairings, draw_prob)

    # R1+R2 only (for validation)
    print("── TrueSkill trained on rounds 1+2 ──")
    ratings_r12  = _run_ts(pairings, draw_prob, max_round=2)

    # DataFrames
    ts_df = _build_df(ratings_full, conn, career)
    vs_df = _build_vs_elo(ts_df, elo_df)

    ts_df.to_csv(data("trueskill_ratings.csv"), index=False)
    vs_df.to_csv(data("trueskill_vs_elo.csv"),  index=False)
    print(f"\nSaved: trueskill_ratings.csv  |  trueskill_vs_elo.csv")

    # Top 15
    print("\nTop 15 by conservative rating:")
    print(ts_df[["nick","full_name","team","mu","sigma",
                 "conservative","total_games"]].head(15).to_string(index=False))

    # Validation
    print("\n── Round 3 Validation ──")
    ts_acc, elo_acc = _validate(conn, ratings_r12, elo_df)

    # Plots
    print("\n── Plots ──")
    _plot_ratings(ts_df)
    _plot_vs_elo(vs_df)

    conn.close()
    return ratings_full, ts_df, ts_acc, elo_acc


if __name__ == "__main__":
    run()
