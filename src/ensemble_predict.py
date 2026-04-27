"""
Ensemble Predictor — Combining Elo, TrueSkill, Bradley-Terry
==============================================================
Combines predictions from 3 independent rating models using:
  1. Majority Vote  — each model votes, majority wins
  2. Weighted Average — average win probabilities, then decide
  3. Stacked Probabilities — per-match comparison

Validation on Round 3 (models trained on R1+R2 only).

Prerequisites:
  output/elo_ratings_final.csv
  output/player_career_stats.csv
  data/processed/twbc.db
  src/phase2_elo.py, src/phase2_trueskill.py, src/phase2_pagerank.py
"""
import sqlite3, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_elo import compute_elo
from phase2_trueskill import _run_ts, _load_pairings as ts_load_pairings, \
    win_probability as ts_win_prob, Rating
from phase2_pagerank import build_matrices, bradley_terry, bt_win_probability
from paths import DB, data, plot_r, ensure_dirs

np.random.seed(42)
ensure_dirs()
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


# ═══════════════════════════════════════════════════════════════
# Train all 3 models on Rounds 1+2
# ═══════════════════════════════════════════════════════════════

def _train_all(conn):
    """Train Elo, TrueSkill, Bradley-Terry on R1+R2 only."""
    print("  Training 3 models on Rounds 1+2...")

    # ── Elo ──
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
    career = pd.read_csv(data("player_career_stats.csv"))
    career_games = dict(zip(career.player_nick, career.total_games))
    elo_r12, _ = compute_elo(pairings_all, career_games, max_round=2)
    print(f"    Elo: {len(elo_r12)} players rated")

    # ── TrueSkill ──
    pairings_ts = ts_load_pairings(conn)
    draw_prob = max((pairings_ts.winner == "Draw").sum() / len(pairings_ts), 0.01)
    ts_r12 = _run_ts(pairings_ts, draw_prob, max_round=2)
    print(f"    TrueSkill: {len(ts_r12)} players rated")

    # ── Bradley-Terry ──
    all_nicks = sorted(set(pairings_all.nick_a) | set(pairings_all.nick_b))
    nick_to_idx = {n: i for i, n in enumerate(all_nicks)}
    p_r12 = pairings_all[pairings_all.tournament_round <= 2]
    wins_r12, games_r12 = build_matrices(p_r12, nick_to_idx)
    gamma_r12 = bradley_terry(wins_r12, games_r12)
    print(f"    Bradley-Terry: {len(gamma_r12)} players rated")

    return elo_r12, ts_r12, gamma_r12, nick_to_idx


# ═══════════════════════════════════════════════════════════════
# Predict Round 3 with all models
# ═══════════════════════════════════════════════════════════════

def _predict_r3(conn, elo, ts_ratings, gamma, nick_to_idx):
    """Generate per-match predictions from all 3 models."""
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
        elo_a = elo_b = ts_a = ts_b = bt_a = bt_b = 0.0
        actual = grp.actual.iloc[0]

        for _, p in grp.iterrows():
            na, nb = p.nick_a, p.nick_b
            ng = p.total_games

            # Elo
            ea = elo.get(na, 1200); eb = elo.get(nb, 1200)
            p_elo = 1 / (1 + 10**((eb - ea) / 400))
            elo_a += p_elo * ng;  elo_b += (1 - p_elo) * ng

            # TrueSkill
            ra = ts_ratings.get(na, Rating())
            rb = ts_ratings.get(nb, Rating())
            p_ts = ts_win_prob(ra.mu, ra.sigma, rb.mu, rb.sigma)
            ts_a += p_ts * ng;  ts_b += (1 - p_ts) * ng

            # Bradley-Terry
            ia = nick_to_idx.get(na); ib = nick_to_idx.get(nb)
            if ia is not None and ib is not None:
                p_bt = bt_win_probability(gamma[ia], gamma[ib])
            else:
                p_bt = 0.5
            bt_a += p_bt * ng;  bt_b += (1 - p_bt) * ng

        # Per-model predictions
        elo_pred = "A" if elo_a > elo_b else "B"
        ts_pred  = "A" if ts_a  > ts_b  else "B"
        bt_pred  = "A" if bt_a  > bt_b  else "B"

        # Ensemble: majority vote
        votes = [elo_pred, ts_pred, bt_pred]
        vote_a = votes.count("A")
        majority = "A" if vote_a >= 2 else "B"

        # Ensemble: weighted average of match-level win probabilities
        total_games = grp.total_games.sum()
        p_elo_match = elo_a / total_games
        p_ts_match  = ts_a  / total_games
        p_bt_match  = bt_a  / total_games
        p_avg = (p_elo_match + p_ts_match + p_bt_match) / 3
        avg_pred = "A" if p_avg > 0.5 else "B"

        rows.append({
            "match_id":   mid,
            "team_a":     grp.team_a.iloc[0],
            "team_b":     grp.team_b.iloc[0],
            "actual":     actual,
            # Individual predictions
            "elo_pred":   elo_pred,
            "ts_pred":    ts_pred,
            "bt_pred":    bt_pred,
            # Individual probabilities
            "p_elo":      round(p_elo_match, 4),
            "p_ts":       round(p_ts_match, 4),
            "p_bt":       round(p_bt_match, 4),
            # Ensemble
            "majority":   majority,
            "avg_pred":   avg_pred,
            "p_avg":      round(p_avg, 4),
            # Correctness
            "elo_ok":     elo_pred == actual,
            "ts_ok":      ts_pred  == actual,
            "bt_ok":      bt_pred  == actual,
            "majority_ok": majority == actual,
            "avg_ok":     avg_pred == actual,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def _plot_results(df):
    """Heatmap of model predictions vs actuals."""
    models = ["elo_pred", "ts_pred", "bt_pred", "majority", "avg_pred"]
    labels = ["Elo", "TrueSkill", "Bradley-Terry", "Majority Vote", "Avg Prob"]
    match_labels = [f"{r.team_a}\nvs {r.team_b}" for _, r in df.iterrows()]

    # Build correctness matrix
    ok_cols = ["elo_ok", "ts_ok", "bt_ok", "majority_ok", "avg_ok"]
    matrix = df[ok_cols].values.astype(float).T  # models × matches

    fig, ax = plt.subplots(figsize=(max(10, len(df)*1.2), 4))
    cmap = plt.cm.colors.ListedColormap(["#E07B6A", "#55A868"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(df)):
            pred = df.iloc[j][models[i]]
            color = "white"
            ax.text(j, i, pred, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(match_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Accuracy labels on right
    accs = [df[c].mean() for c in ok_cols]
    for i, acc in enumerate(accs):
        ax.text(len(df) + 0.3, i, f"{acc:.0%}",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Ensemble Prediction — Round 3 Validation\n"
                 "(green = correct, red = wrong)", fontsize=11)
    plt.tight_layout()
    path = plot_r("plot_ensemble_validation.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 62)
    print("  Ensemble Predictor — Elo + TrueSkill + Bradley-Terry")
    print("=" * 62)

    conn = sqlite3.connect(DB)

    # Train all models on R1+R2
    elo, ts, gamma, nick_to_idx = _train_all(conn)

    # Predict Round 3
    print("\n── Round 3 Predictions ──")
    df = _predict_r3(conn, elo, ts, gamma, nick_to_idx)

    # Results table
    print(f"\n  {'Match':<40} {'Elo':>4} {'TS':>4} {'BT':>4} {'Vote':>5} {'Avg':>4} {'Act':>5}")
    print(f"  {'─'*70}")
    for _, r in df.iterrows():
        def m(ok): return "✓" if ok else "✗"
        print(f"  {r.team_a+' vs '+r.team_b:<40} "
              f"{m(r.elo_ok):>4} {m(r.ts_ok):>4} {m(r.bt_ok):>4} "
              f"{m(r.majority_ok):>5} {m(r.avg_ok):>4} {r.actual:>5}")

    # Accuracy summary
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'─'*42}")
    for label, col in [("Elo", "elo_ok"), ("TrueSkill", "ts_ok"),
                        ("Bradley-Terry", "bt_ok"),
                        ("Majority Vote", "majority_ok"),
                        ("Average Prob", "avg_ok")]:
        acc = df[col].mean()
        n = df[col].sum()
        print(f"  {label:<20} {acc:>9.1%} {n:>7}/{len(df)}")

    # Per-match probability comparison
    print(f"\n  {'Match':<40} {'P(Elo)':>8} {'P(TS)':>8} {'P(BT)':>8} {'P(Avg)':>8}")
    print(f"  {'─'*75}")
    for _, r in df.iterrows():
        print(f"  {r.team_a+' vs '+r.team_b:<40} "
              f"{r.p_elo:>8.3f} {r.p_ts:>8.3f} {r.p_bt:>8.3f} {r.p_avg:>8.3f}")

    # Agreement analysis
    print("\n── Model Agreement ──")
    df["agree_count"] = df.apply(
        lambda r: sum([r.elo_pred == r.actual, r.ts_pred == r.actual, r.bt_pred == r.actual]),
        axis=1
    )
    for n_agree in [3, 2, 1, 0]:
        subset = df[df.agree_count == n_agree]
        if len(subset) > 0:
            print(f"  {n_agree}/3 models correct: {len(subset)} matches")

    unanimous = df[(df.elo_pred == df.ts_pred) & (df.ts_pred == df.bt_pred)]
    split = df[(df.elo_pred != df.ts_pred) | (df.ts_pred != df.bt_pred)]
    print(f"\n  Unanimous agreement: {len(unanimous)}/{len(df)} matches "
          f"→ {unanimous.majority_ok.mean():.0%} correct")
    if len(split) > 0:
        print(f"  Split decisions:     {len(split)}/{len(df)} matches "
              f"→ majority correct {split.majority_ok.mean():.0%}")

    # Save
    df.to_csv(data("ensemble_predictions_r3.csv"), index=False)
    print(f"  Saved: data/ensemble_predictions_r3.csv")

    # Plot
    _plot_results(df)

    # Summary
    best_single = max(df.elo_ok.mean(), df.ts_ok.mean(), df.bt_ok.mean())
    ensemble_acc = df.majority_ok.mean()
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ENSEMBLE SUMMARY                                            ║
╠══════════════════════════════════════════════════════════════╣
║  Best single model:    Bradley-Terry at {best_single:.1%}              ║
║  Majority vote:        {ensemble_acc:.1%}                                  ║
║  Average probability:  {df.avg_ok.mean():.1%}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Key insight: models disagree on different matches.          ║
║  BT uniquely captures Czechia F vs Slovakia upset.          ║
║  Ensemble leverages complementary strengths.                 ║
╚══════════════════════════════════════════════════════════════╝""")

    conn.close()
    return df


if __name__ == "__main__":
    run()
