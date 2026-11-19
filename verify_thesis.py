"""
Check every quantitative claim in the thesis against what the code actually produces.

Usage : python verify_thesis.py                     (uses experiments/outputs, shipped with the package)
        python verify_thesis.py --outputs DIR       (uses freshly regenerated experiment outputs)

Two kinds of evidence are checked:
  * the saved models in models/ are re-scored live on the 200 test applicants, and the SMOTE
    finding of Section 5.8 is recomputed from scratch;
  * every other number in Chapters 5 and 6 and Appendix A is compared with results.json and
    supplementary.json, written by the two scripts in experiments/.

Writes thesis_check_report.md (and .csv) and exits with status 1 if any claim fails.
"""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
import credit_model as cm

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=str(cm.HERE / "experiments" / "outputs"))
args = ap.parse_args()
OUT = Path(args.outputs).resolve()
R = json.loads((OUT / "results.json").read_text())
S = json.loads((OUT / "supplementary.json").read_text())
LR, DT = cm.LR, cm.DT

rows = []


def check(where, claim, thesis, actual, tol=0.0005):
    if isinstance(thesis, str):
        ok = str(actual) == thesis
    else:
        ok = actual is not None and abs(float(actual) - float(thesis)) <= tol
    rows.append((where, claim, thesis, actual if isinstance(actual, str) else round(float(actual), 4), ok))


# ----------------------------------------------------------------------------- live checks on the saved models
X, y = cm.load_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=cm.SEED)
models, card = cm.load_models()
P = {n: m.predict(Xte) for n, m in models.items()}
Sc = {n: m.predict_proba(Xte)[:, 1] for n, m in models.items()}

for n, exp in {LR: (0.755, 0.568, 0.767, 0.750, 0.652, 0.809, 0.664, 105),
               DT: (0.655, 0.438, 0.533, 0.707, 0.481, 0.685, 0.437, 181)}.items():
    p, s, t = P[n], Sc[n], yte.values
    got = (accuracy_score(t, p), precision_score(t, p), recall_score(t, p), recall_score(t, p, pos_label=0),
           f1_score(t, p), roc_auc_score(t, s), average_precision_score(t, s),
           5 * ((t == 1) & (p == 0)).sum() + ((t == 0) & (p == 1)).sum())
    for name, e, g in zip(["accuracy", "precision", "recall", "specificity", "F1", "AUC", "AP", "cost"], exp, got):
        check("Table 5.4 (saved model, live)", f"{n} {name}", e, g, 0.5 if name == "cost" else 0.0005)

c_lr, c_dt = P[LR] == yte.values, P[DT] == yte.values
b, c = int((c_lr & ~c_dt).sum()), int((~c_lr & c_dt).sum())
check("Table 5.6 (live)", "LR right, tree wrong", 31, b)
check("Table 5.6 (live)", "tree right, LR wrong", 11, c)
check("Table 5.6 (live)", "both right", 120, int((c_lr & c_dt).sum()))
check("Section 5.6.1 (live)", "exact McNemar p", 0.003, stats.binomtest(min(b, c), b + c, 0.5).pvalue, 0.0005)

tree = models[DT].named_steps["clf"]
check("Section 5.4 (live)", "pruned tree depth", 6, tree.get_depth())
check("Section 5.4 (live)", "pruned tree leaves", 15, tree.get_n_leaves())
check("Section 5.4 (live)", "inputs used by the tree", 11, int((tree.feature_importances_ > 0).sum()))
check("Section 5.5 (live)", "tree F1 on training data", 0.644, f1_score(ytr, models[DT].predict(Xtr)), 0.0005)
check("Section 5.2 (live)", "duplicate records", 0, int(X.assign(t=y).duplicated().sum()))

# Section 5.8: synthetic records with impossible fractional dummies, and the leaves they fill
pre = models[DT].named_steps["pre"]
Z = pre.transform(Xtr)
Zs, _ = SMOTE(random_state=cm.SEED).fit_resample(Z, ytr.values)
syn = Zs[len(Z):, len(cm.NUM):]
check("Section 5.3 (live)", "synthetic defaulters added by SMOTE", 320, len(syn))
check("Section 5.8 (live)", "synthetic records with a fractional dummy", 320, int(((syn > 1e-9) & (syn < 1 - 1e-9)).any(1).sum()))
leaf_real, leaf_syn = tree.apply(Zs[:len(Z)]), tree.apply(Zs[len(Z):])
only_syn = sorted([int((leaf_syn == l).sum()) for l in np.unique(leaf_syn) if (leaf_real == l).sum() == 0], reverse=True)
check("Section 5.8 (live)", "records in the synthetic-only leaves", "80 + 9", " + ".join(map(str, only_syn)))

# Section 5.5: matched operating points
fpr, tpr, thr = roc_curve(yte, Sc[LR])
spec_dt, rec_dt = recall_score(yte, P[DT], pos_label=0), recall_score(yte, P[DT])
check("Section 5.5 (live)", "LR recall at the tree's specificity", 0.767, tpr[(1 - fpr) >= spec_dt - 1e-9].max(), 0.0005)
check("Section 5.5 (live)", "LR specificity at the tree's recall", 0.893, (1 - fpr)[tpr >= rec_dt - 1e-9].max(), 0.0005)

# Appendix A.2: applicant T115
import pandas as pd
t115 = pd.read_csv(cm.HERE / "examples" / "sample_applicants.csv").set_index("applicant_id").loc["T115"].drop("actual_outcome").to_dict()
res = cm.score(models, card, t115)
check("Table A.2 (live)", "T115 LR risk score", 0.502, res[LR]["p_bad"], 0.0005)
check("Table A.2 (live)", "T115 tree risk score", 0.156, res[DT]["p_bad"], 0.0005)
check("Table A.2 (live)", "T115 LR decision at 0.29", "REFER / DECLINE", res[LR]["decision_cost"])
check("Table A.2 (live)", "T115 tree decision at 0.5", "APPROVE", res[DT]["decision_default"])
check("Appendix A.2 (live)", "T115 LR score with no checking account", 0.183,
      cm.score(models, card, dict(t115, checking_status="no checking account"))[LR]["p_bad"], 0.0005)
tab, _, _ = cm.explain_lr(models[LR], t115)
contrib = dict(zip(tab.attribute, tab.contribution))
for attr, v in [("checking_status", 0.52), ("foreign_worker", 0.48), ("savings", 0.35),
                ("other_installment_plans", -0.69), ("employment", -0.58)]:
    check("Table A.2 (live)", f"T115 LR contribution of {attr}", v, contrib[attr], 0.005)

# ----------------------------------------------------------------------------- experiment outputs
check("Section 5.2", "records", 1000, R["n"]); check("Section 5.2", "missing values", 0, R["missing"])
check("Section 5.2", "defaulters", 300, R["class_counts"]["bad"])
for attr, (mean, sd, mn, med, mx) in {"duration": (20.9, 12.1, 4, 18, 72), "credit_amount": (3271.3, 2822.7, 250, 2319.5, 18424),
                                      "age": (35.5, 11.4, 19, 33, 75)}.items():
    d = R["describe"][attr]
    for k, e in zip(["mean", "std", "min", "50%", "max"], (mean, sd, mn, med, mx)):
        check("Table 5.1", f"{attr} {k}", e, d[k], 0.05)
for cat, val in [("no checking account", 11.7), ("< 0 DM", 49.3), ("0-200 DM", 39.0), (">= 200 DM / salary", 22.2)]:
    check("Table 5.2", f"default rate, checking account {cat}", val, R["crosstab"]["checking_status"][cat], 0.05)
for cat, val in [("critical account", 17.1), ("all paid at this bank", 57.1), ("no credits / all paid", 62.5)]:
    check("Table 5.2", f"default rate, credit history {cat}", val, R["crosstab"]["credit_history"][cat], 0.05)
for cat, val in [("own", 26.1), ("rent", 39.1), ("for free", 40.7)]:
    check("Table 5.2", f"default rate, housing {cat}", val, R["crosstab"]["housing"][cat], 0.05)

check("Section 5.3", "encoded inputs", 48, R["n_encoded"])
check("Section 5.3", "largest correlation |r|", 0.74, R["corr_max_pair"][2], 0.005)
check("Section 5.3", "pairs above 0.85", 0, R["corr_over_085"])
check("Section 5.3", "records per class after SMOTE", 560, R["after_smote"]["bad"])
check("Table 5.3", "LR selected C", 0.1, R["tuning"][LR]["params"]["C"])
check("Table 5.3", "LR CV F1 (validation)", 0.599, R["tuning"][LR]["cv_f1"], 0.0005)
check("Table 5.3", "LR CV F1 (training folds)", 0.637, R["tuning"][LR]["train_f1"], 0.0005)
check("Table 5.3", "tree CV F1 (validation)", 0.574, R["tuning"][DT]["cv_f1"], 0.0005)
check("Table 5.3", "tree CV F1 (training folds)", 0.678, R["tuning"][DT]["train_f1"], 0.0005)
check("Table 3.1", "tree grid size", 896, R["tuning"][DT]["n_configs"])
check("Section 5.4", "validation F1 at depth 1 (curve peak)", 0.558, R["validation_curve"]["cv"][0], 0.0005)

u = R["test"]["Unconstrained Tree"]
for k, e in [("accuracy", 0.690), ("recall", 0.550), ("f1", 0.516), ("auc", 0.650), ("cost", 170)]:
    check("Table 5.4", f"unpruned tree {k}", e, u[k], 0.5 if k == "cost" else 0.0005)
check("Table 5.4", "majority baseline accuracy", 0.700, R["test"]["Majority baseline"]["accuracy"])
check("Table 5.4", "majority baseline cost", 300, R["test"]["Majority baseline"]["cost"])
for n, cls, k, e in [(LR, "good", "precision", 0.882), (LR, "good", "f1", 0.811), (DT, "good", "precision", 0.780), (DT, "good", "f1", 0.742)]:
    check("Table 5.5", f"{n} {cls} risk {k}", e, S["by_class"][n][cls][k], 0.0005)

check("Section 5.6.1", "McNemar odds ratio", 2.82, S["mcnemar_or"]["or"], 0.005)
check("Section 5.6.1", "odds ratio CI lower", 1.38, S["mcnemar_or"]["lo"], 0.005)
check("Section 5.6.1", "odds ratio CI upper", 6.21, S["mcnemar_or"]["hi"], 0.005)
for m, (lr, dt, wins, t, d, F, p5) in {"f1": (0.608, 0.501, 93, 4.21, 1.47, 8.90, 0.013), "auc": (0.789, 0.717, 94, 4.77, 1.66, 16.38, 0.003),
                                       "acc": (0.724, 0.695, 72, 1.82, 0.63, 21.95, 0.002)}.items():
    check("Table 5.7", f"{m}: LR mean", lr, R["repeated_cv"][LR][m][0], 0.0005)
    check("Table 5.7", f"{m}: tree mean", dt, R["repeated_cv"][DT][m][0], 0.0005)
    check("Table 5.7", f"{m}: LR wins of 100", wins, S["corrected_cv"][m]["wins"])
    check("Table 5.7", f"{m}: corrected t", t, S["corrected_cv"][m]["t"], 0.005)
    check("Table 5.7", f"{m}: effect size d", d, S["corrected_cv"][m]["d"], 0.005)
    check("Table 5.7", f"{m}: 5x2cv F", F, S["f_5x2"][m]["F"], 0.005)
    check("Table 5.7", f"{m}: 5x2cv p", p5, S["f_5x2"][m]["p"], 0.0005)
check("Table 5.7", "accuracy: corrected t-test p", 0.072, S["corrected_cv"]["acc"]["p"], 0.0005)
check("Section 5.6.2", "tree F1 SD across folds", 0.080, R["repeated_cv"][DT]["f1"][1], 0.0005)
check("Section 5.6.2", "LR F1 SD across folds", 0.054, R["repeated_cv"][LR]["f1"][1], 0.0005)
for m, (dlt, lo, hi) in {"recall": (0.233, 0.133, 0.35), "f1": (0.171, 0.093, 0.261), "auc": (0.124, 0.057, 0.191),
                         "specificity": (0.043, -0.029, 0.114), "cost": (-76, -110, -43)}.items():
    tol = 0.5 if m == "cost" else 0.0005
    check("Figure 5.5", f"bootstrap {m} difference", dlt, S["bootstrap"][m]["diff"], tol)
    check("Figure 5.5", f"bootstrap {m} CI lower", lo, S["bootstrap"][m]["lo"], tol + 0.005)
    check("Figure 5.5", f"bootstrap {m} CI upper", hi, S["bootstrap"][m]["hi"], tol + 0.005)
for m, (dlt, lo, hi) in {"f1": (0.107, 0.056, 0.157), "auc": (0.072, 0.042, 0.102), "acc": (0.029, -0.003, 0.060)}.items():
    check("Figure 5.5", f"CV {m} difference", dlt, S["corrected_cv"][m]["mean_diff"], 0.0005)
    check("Figure 5.5", f"CV {m} CI", f"{lo} to {hi}", f"{S['corrected_cv'][m]['lo']} to {S['corrected_cv'][m]['hi']}")

th = R["threshold"]
check("Section 5.7", "LR cost-based threshold", 0.29, th[LR]["t"]); check("Section 5.7", "tree cost-based threshold", 0.09, th[DT]["t"])
check("Section 5.7", "LR out-of-fold cost at 0.5", 513, th[LR]["oof_cost_default"]); check("Section 5.7", "LR out-of-fold cost at 0.29", 422, th[LR]["oof_cost_t"])
for k, e in [("recall", 0.883), ("precision", 0.414), ("specificity", 0.464), ("f1", 0.564), ("cost", 110)]:
    check("Table 5.8", f"LR at 0.29 {k}", e, th[LR]["test"][k], 0.5 if k == "cost" else 0.0005)
check("Table 5.8", "tree at 0.09 cost (refuses everyone)", 140, th[DT]["test"]["cost"])
for ratio, (lrc, dtc, appr) in {"1": (44, 69, 60), "2": (69, 97, 120), "5": (110, 140, 300), "10": (114, 140, 600)}.items():
    cr = S["cost_ratio"][ratio]
    check("Table 5.9", f"{ratio}:1 LR cost", lrc, cr[LR]["cost"]); check("Table 5.9", f"{ratio}:1 tree cost", dtc, cr[DT]["cost"])
    check("Table 5.9", f"{ratio}:1 approve-all cost", appr, cr["approve_all"]); check("Table 5.9", f"{ratio}:1 refuse-all cost", 140, cr["reject_all"])

rob = S["robustness"]
trees = {k: v for k, v in rob.items() if k.startswith("DT")}
gaps = [v["vs_lr_auc"]["mean_diff"] for v in trees.values()]
check("Table 5.10", "tree variants", 10, len(trees)); check("Table 5.10", "LR variants", 6, len(rob) - len(trees))
check("Section 5.8", "smallest LR - tree CV AUC gap", 0.063, min(gaps)); check("Section 5.8", "largest LR - tree CV AUC gap", 0.106, max(gaps))
check("Section 5.8", "every tree variant significantly behind (max p)", 0.0, max(v["vs_lr_auc"]["p"] for v in trees.values()), 0.001)
check("Section 5.8", "best tree CV AUC", 0.725, max(v["cv_auc"] for v in trees.values()))
best = rob["DT: class weighting, ordinal encoding"]
check("Section 5.8", "best tree", "DT: class weighting, ordinal encoding", S["best_dt_variant"])
for k, e in [("recall", 0.733), ("auc", 0.730), ("cost", 121)]:
    check("Section 5.8", f"ordinal tree test {k}", e, best["test"][k], 0.5 if k == "cost" else 0.0005)
check("Section 5.8", "ordinal tree size", "3, 8", ", ".join(map(str, best["depth_leaves"])))
check("Section 5.8", "ordinal tree CV F1", 0.565, best["cv_f1"])
check("Section 5.8", "LR without imbalance treatment: specificity", 0.879, rob["LR: no imbalance treatment"]["test"]["specificity"])
check("Section 5.8", "LR without imbalance treatment: AUC", 0.805, rob["LR: no imbalance treatment"]["test"]["auc"])

coef = {c[0]: c for c in R["lr_coef"]}
check("Table 5.11", "no checking account odds ratio", 0.37, coef["checking_status_no checking account"][2])
check("Table 5.11", "overdrawn account odds ratio", 1.68, coef["checking_status_< 0 DM"][2])
check("Table 5.11", "foreign worker odds ratio", 1.62, coef["foreign_worker_yes"][2])
check("Table 5.11", "male single odds ratio", 0.59, coef["personal_status_male single"][2])
gini = dict(R["dt_gini"])
check("Section 5.9.2", "Gini importance of no checking account", 0.411, gini["checking_status_no checking account"])
check("Section 5.9.2", "Gini importance of duration", 0.124, gini["duration"])
perm = {n: {a: v for a, v, _ in R["perm"][n]} for n in (LR, DT)}
check("Section 5.9.3", "LR permutation importance, checking account", 0.098, perm[LR]["checking_status"], 0.0005)
check("Section 5.9.3", "tree permutation importance, checking account", 0.162, perm[DT]["checking_status"], 0.0005)
check("Section 5.9.3", "LR permutation importance, purpose", 0.033, perm[LR]["purpose"], 0.0005)
check("Section 5.9.3", "fairness-sensitive attributes each below 0.01 for LR", 1,
      int(all(perm[LR].get(a, 0) < 0.01 for a in ("age", "personal_status", "foreign_worker"))))

# ----------------------------------------------------------------------------- report
passed = sum(r[4] for r in rows)
lines = ["# Thesis verification report", "",
         f"**{passed} of {len(rows)} claims reproduced.** Rows marked *(live)* were recomputed just now from the "
         "saved models; the rest come from the experiment outputs in `" + (str(OUT.relative_to(cm.HERE)) if OUT.is_relative_to(cm.HERE) else str(OUT)) + "`.", "",
         "| Where in the thesis | Claim | Thesis | Code | Result |", "|---|---|---|---|---|"]
lines += [f"| {w} | {c} | {t} | {a} | {'PASS' if ok else '**FAIL**'} |" for w, c, t, a, ok in rows]
(cm.HERE / "thesis_check_report.md").write_text("\n".join(lines) + "\n")
pd.DataFrame(rows, columns=["where", "claim", "thesis", "code", "passed"]).to_csv(cm.HERE / "thesis_check_report.csv", index=False)
for w, c, t, a, ok in rows:
    if not ok:
        print(f"FAIL  {w}: {c}  thesis {t}  code {a}")
print(f"{passed}/{len(rows)} thesis claims reproduced. Full table in thesis_check_report.md")
sys.exit(0 if passed == len(rows) else 1)
