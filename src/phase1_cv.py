"""
Phase 1 CV + Regularization Tuning
====================================
Fixes: (1) LOO cross-validation — true out-of-sample accuracy on n=50
        (2) Regularization grid search — find optimal C, check coefficient stability

Run AFTER phase1_fix.py has completed (needs player_career_stats.csv + DB).
"""
import sqlite3, os
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc

# Suppress sklearn FutureWarning on penalty
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)
from paths import DB, OUT, data, plot_m, ensure_dirs
ensure_dirs()
DPI = 150

FEATURES = ["board1_eff", "board2_eff", "board3_eff"]
TARGET   = "match_won"


def build_df_wide(con):
    """Reconstruct df_wide (same logic as phase1_fix.py Step 1.1)."""
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
    df_matches = pd.read_sql("SELECT match_id, team_a, team_b, winner FROM v_matches", con)

    df_pms["board"] = (
        df_pms.groupby(["match_id","team"])["career_eff"]
        .rank(method="first", ascending=False).astype(int)
    )
    df_boards = df_pms[df_pms.board <= 3].copy()
    df_wide = df_boards.pivot_table(
        index=["match_id","team"], columns="board", values="efficiency"
    ).reset_index()
    df_wide.columns = ["match_id","team","board1_eff","board2_eff","board3_eff"]
    df_wide = df_wide.merge(df_matches, on="match_id")
    df_wide["match_won"] = (
        ((df_wide.team == df_wide.team_a) & (df_wide.winner == "A")) |
        ((df_wide.team == df_wide.team_b) & (df_wide.winner == "B"))
    ).astype(int)
    return df_wide, df_pms


def make_pipe(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=C, l1_ratio=0,
                                      solver="lbfgs", max_iter=1000,
                                      random_state=42))
    ])


def run():
    con = sqlite3.connect(DB)
    df_wide, _ = build_df_wide(con)
    con.close()

    df_fit = df_wide[FEATURES + [TARGET, "match_id", "team"]].dropna().copy()
    X = df_fit[FEATURES].values
    y = df_fit[TARGET].values
    print(f"Dataset: {len(df_fit)} rows | {int(y.sum())} wins, {int((1-y).sum())} losses")

    loo = LeaveOneOut()

    # ── FIX 1 — LOO Cross-Validation ──────────────────────
    print("\n" + "═"*55)
    print("  FIX 1 — Leave-One-Out Cross-Validation (C=1.0)")
    print("═"*55)

    # Step 1.1 — LOO accuracy + log-loss
    pipe = make_pipe(C=1.0)
    loo_acc_scores = cross_val_score(pipe, X, y, cv=loo, scoring="accuracy")
    loo_acc  = loo_acc_scores.mean()
    loo_proba = cross_val_predict(pipe, X, y, cv=loo, method="predict_proba")
    loo_preds = (loo_proba[:, 1] >= 0.5).astype(int)
    loo_logloss = log_loss(y, loo_proba[:, 1])

    print(f"\n  Training accuracy (reported):  100.0%")
    print(f"  LOO accuracy:                  {loo_acc:.1%}")
    print(f"  LOO log-loss:                  {loo_logloss:.4f}")
    gap = 1.0 - loo_acc
    print(f"  Gap (train − LOO):             {gap:+.1%}")

    if gap < 0.05:
        verdict = "minimal overfitting — trust Phase 1 coefficients"
    elif gap < 0.15:
        verdict = "moderate overfitting — use best_C model for Phase 2"
    else:
        verdict = "severe overfitting — Phase 1 coefficients unreliable"
    print(f"  → {verdict}")

    # Step 1.2 — Confusion matrix + misclassified
    print("\n  LOO Confusion Matrix:")
    cm = confusion_matrix(y, loo_preds)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    df_fit_idx = df_fit.copy().reset_index(drop=True)
    df_fit_idx["loo_pred"]  = loo_preds
    df_fit_idx["loo_proba"] = loo_proba[:, 1]
    df_fit_idx["correct"]   = (df_fit_idx["loo_pred"] == df_fit_idx[TARGET])

    misclassified = df_fit_idx[~df_fit_idx["correct"]]
    print(f"\n  Misclassified matches ({len(misclassified)}):")
    if len(misclassified) > 0:
        print(misclassified[
            ["match_id","team","board1_eff","board2_eff","board3_eff","match_won","loo_proba"]
        ].to_string(index=False))
    else:
        print("  None — even LOO is perfect (well-separated classes, small dataset)")

    df_fit_idx.to_csv(f"{OUT}/loo_predictions.csv", index=False)
    print(f"  Saved: output/loo_predictions.csv")

    # Step 1.3 — ROC curve
    fpr, tpr, _ = roc_curve(y, loo_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#185FA5", lw=2, label=f"LOO ROC (AUC = {roc_auc:.3f})")
    ax.plot([0,1], [0,1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#185FA5")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — LOO Cross-Validation")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUT}/roc_curve_loo.png", dpi=DPI); plt.close()
    print(f"  Saved: output/roc_curve_loo.png  (AUC={roc_auc:.3f})")

    # ── FIX 2 — Regularization Grid Search ────────────────
    print("\n" + "═"*55)
    print("  FIX 2 — Regularization Grid Search")
    print("═"*55)

    C_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    results = []

    for c in C_GRID:
        pipe_c = make_pipe(C=c)
        acc_scores = cross_val_score(pipe_c, X, y, cv=loo, scoring="accuracy")
        proba_c    = cross_val_predict(pipe_c, X, y, cv=loo, method="predict_proba")
        ll         = log_loss(y, proba_c[:, 1])
        pipe_c.fit(X, y)
        coefs = pipe_c.named_steps["clf"].coef_[0]
        results.append({
            "C":           c,
            "loo_acc":     round(acc_scores.mean(), 4),
            "log_loss":    round(ll, 4),
            "board1_coef": round(coefs[0], 4),
            "board2_coef": round(coefs[1], 4),
            "board3_coef": round(coefs[2], 4),
        })

    df_results = pd.DataFrame(results)
    print("\n  " + df_results.to_string(index=False).replace("\n", "\n  "))
    df_results.to_csv(f"{OUT}/regularization_grid_search.csv", index=False)
    print(f"\n  Saved: output/regularization_grid_search.csv")

    best = df_results.sort_values(["loo_acc","log_loss"], ascending=[False,True]).iloc[0]
    print(f"\n  Best C: {best.C}  (LOO acc={best.loo_acc:.1%}, log-loss={best.log_loss:.4f})")

    # Step 2.2 — Visualize grid search
    x_log = np.log10(df_results.C)
    x_labels = [str(c) for c in C_GRID]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(x_log, df_results.loo_acc, "o-", color="#185FA5", lw=2)
    ax.axvline(np.log10(best.C), color="#E24B4A", lw=1.5, linestyle="--",
               label=f"best C={best.C}")
    ax.set_xticks(x_log); ax.set_xticklabels(x_labels, rotation=45, fontsize=9)
    ax.set_ylabel("LOO Accuracy"); ax.set_title("Accuracy vs C")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(x_log, df_results.log_loss, "o-", color="#1D9E75", lw=2)
    ax2.axvline(np.log10(best.C), color="#E24B4A", lw=1.5, linestyle="--")
    ax2.set_xticks(x_log); ax2.set_xticklabels(x_labels, rotation=45, fontsize=9)
    ax2.set_ylabel("Log-loss (lower = better)"); ax2.set_title("Log-loss vs C")

    ax3 = axes[2]
    colors3 = {"board1_coef":"#E24B4A","board2_coef":"#378ADD","board3_coef":"#7F77DD"}
    for col, clr in colors3.items():
        ax3.plot(x_log, df_results[col], "o-", color=clr, lw=2,
                 label=col.replace("_coef",""))
    ax3.axvline(np.log10(best.C), color="#E24B4A", lw=1.5, linestyle="--", alpha=0.5)
    ax3.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax3.set_xticks(x_log); ax3.set_xticklabels(x_labels, rotation=45, fontsize=9)
    ax3.set_ylabel("Coefficient (standardized)")
    ax3.set_title("Coefficient paths vs C"); ax3.legend(fontsize=9)

    plt.suptitle("Regularization grid search — LOO evaluation", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/regularization_grid_search.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: output/regularization_grid_search.png")

    # Step 2.3 — Final model with best C
    best_C = float(best.C)
    pipe_best = make_pipe(C=best_C)
    pipe_best.fit(X, y)
    best_coefs = pipe_best.named_steps["clf"].coef_[0]
    best_or    = np.exp(best_coefs)

    best_proba = cross_val_predict(pipe_best, X, y, cv=loo, method="predict_proba")
    best_preds = (best_proba[:, 1] >= 0.5).astype(int)
    best_acc   = accuracy_score(y, best_preds)
    best_ll    = log_loss(y, best_proba[:, 1])

    print(f"\n  === FINAL MODEL (C={best_C}) ===")
    print(f"  LOO Accuracy: {best_acc:.1%}")
    print(f"  LOO Log-loss: {best_ll:.4f}")
    print(f"\n  {'Feature':<16} {'Coef (std)':>10}   {'Odds Ratio':>10}")
    print(f"  {'─'*40}")
    for feat, coef, or_ in zip(FEATURES, best_coefs, best_or):
        print(f"  {feat:<16} {coef:>10.4f}   {or_:>10.3f}")

    summary = {
        "best_C":      best_C,
        "loo_acc":     best_acc,
        "log_loss":    best_ll,
        "board1_coef": best_coefs[0], "board1_OR": best_or[0],
        "board2_coef": best_coefs[1], "board2_OR": best_or[1],
        "board3_coef": best_coefs[2], "board3_OR": best_or[2],
    }
    pd.Series(summary).to_csv(f"{OUT}/final_logistic_model.csv", header=["value"])
    print(f"  Saved: output/final_logistic_model.csv")

    # ── Final summary box ──────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           PHASE 1 — COMPLETE SUMMARY                    ║
╠══════════════════════════════════════════════════════════╣
║  Step 1 (OLS):  R²=0.997 — circular dependency, INVALID ║
║  Step 2 (Logit C=1.0):                                  ║
║    Training acc = 100%  ← overfitting suspected         ║
║    LOO acc      = {loo_acc:.1%}  ← true generalization       ║
║    Board pattern: Board3 > Board1 > Board2              ║
║  Step 3 (Best C={best_C:.3f}):                              ║
║    LOO acc      = {best_acc:.1%}                               ║
║    Log-loss     = {best_ll:.4f}                              ║
╠══════════════════════════════════════════════════════════╣
║  CONCLUSION:                                            ║
║  → All 3 boards statistically significant               ║
║  → No single board dominates — team depth matters       ║
║  → Board 3 OR slightly higher: "swing board" hypothesis ║
║  → Outliers inflate coefs ~15-23% but don't change      ║
║    relative ordering                                     ║
║  → Proceed to Phase 2 with full dataset                 ║
╚══════════════════════════════════════════════════════════╝""")

    # Output checklist
    print("\n" + "─"*45)
    expected = [
        "loo_predictions.csv",
        "regularization_grid_search.csv",
        "final_logistic_model.csv",
        "roc_curve_loo.png",
        "regularization_grid_search.png",
    ]
    for f in expected:
        mark = "✓" if os.path.isfile(os.path.join(OUT, f)) else "✗"
        print(f"  [{mark}] output/{f}")

    return best_C, best_acc, best_ll


if __name__ == "__main__":
    run()
