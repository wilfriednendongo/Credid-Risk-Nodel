"""
Train the two credit risk models exactly as in the thesis and save them for demonstration.

Usage : python train_and_save.py [path/to/german.data]

Steps : stratified 80/20 split (seed 42) -> grid search with 5-fold CV on F1 for Bad Risk
        -> cost-based threshold chosen on out-of-fold training predictions (5:1 cost ratio)
        -> one evaluation on the untouched test set -> save models and a model card.
Writes: models/logistic_regression.joblib, models/decision_tree.joblib, models/model_card.json,
        examples/sample_applicants.csv (ten test-set applicants with their real outcome)
Time  : about one minute on a four-core laptop.
"""
import sys, json, platform, warnings
from datetime import date
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import sklearn, imblearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
import credit_model as cm

# Load and decode german.data (see credit_model.load_data), then split it 80/20.
# stratify=y keeps 30% defaulters in both parts; the 200 test applicants are only used once, at the end.
data_path = sys.argv[1] if len(sys.argv) > 1 else cm.DATA_FILE
X, y = cm.load_data(data_path)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=cm.SEED)
print(f"Data: {len(X)} applicants, {y.mean():.0%} bad risk. Train {len(Xtr)}, test {len(Xte)}.")


# All test measures for one model. Cost = 5 x missed defaulters + 1 x refused good applicants.
def metrics(yt, p, s):
    return {"accuracy": accuracy_score(yt, p), "precision": precision_score(yt, p, zero_division=0),
            "recall": recall_score(yt, p), "specificity": recall_score(yt, p, pos_label=0),
            "f1": f1_score(yt, p), "auc": roc_auc_score(yt, s), "ap": average_precision_score(yt, s),
            "cost": int(cm.COST_FN * ((yt == 1) & (p == 0)).sum() + cm.COST_FP * ((yt == 0) & (p == 1)).sum()),
            "confusion_matrix [[TP, FN], [FP, TN]]": confusion_matrix(yt, p, labels=[1, 0]).tolist()}


# 5-fold cross-validation, and the thresholds 0.05 to 0.95 tried when picking the cost-based cut-off.
cv5 = StratifiedKFold(5, shuffle=True, random_state=cm.SEED)
grid_t = np.round(np.arange(0.05, 0.951, 0.01), 2)
# The model card records settings, versions and results; it is saved next to the models.
card = {"title": "Credit risk models: Logistic Regression and Decision Tree (Statlog German Credit)",
        "author": "Wilfried Mweneni Nendongo (201304914), University of Namibia",
        "trained_on": str(date.today()), "python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
        "positive_class": "1 = Bad Risk (defaulter)", "cost_ratio": "false negative 5 : false positive 1",
        "train_size": len(Xtr), "test_size": len(Xte), "params": {}, "cv_f1": {}, "thresholds": {},
        "test_default_threshold": {}, "test_cost_threshold": {}}

# Train each model in turn.
for name, (clf, grid) in cm.GRIDS.items():
    # Try every setting in the grid with 5-fold CV on the training data, keep the best by F1.
    gs = GridSearchCV(cm.pipe(clf), grid, scoring="f1", cv=cv5, n_jobs=-1).fit(Xtr, ytr)
    model = gs.best_estimator_
    # Out-of-fold risk scores on the training data, used to choose the threshold without
    # touching the test set. The threshold with the lowest 5:1 cost wins.
    oof = cross_val_predict(clone(model), Xtr, ytr, cv=cv5, method="predict_proba")[:, 1]
    costs = [cm.COST_FN * ((ytr == 1) & (oof < t)).sum() + cm.COST_FP * ((ytr == 0) & (oof >= t)).sum() for t in grid_t]
    t_star = float(grid_t[int(np.argmin(costs))])
    # The one and only evaluation on the untouched test set.
    s = model.predict_proba(Xte)[:, 1]
    card["params"][name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    card["cv_f1"][name] = round(float(gs.best_score_), 4)
    card["thresholds"][name] = t_star
    card["test_default_threshold"][name] = {k: (round(float(v), 4) if isinstance(v, float) else v)
                                            for k, v in metrics(yte.values, (s >= 0.5).astype(int), s).items()}
    card["test_cost_threshold"][name] = {k: (round(float(v), 4) if isinstance(v, float) else v)
                                         for k, v in metrics(yte.values, (s >= t_star).astype(int), s).items()}
    # Save the fitted pipeline (preprocessing + SMOTE + model) to models/.
    joblib.dump(model, cm.MODEL_DIR / cm.MODEL_FILES[name])
    r = card["test_default_threshold"][name]
    print(f"{name:20s} best {card['params'][name]}  test recall {r['recall']:.3f}  AUC {r['auc']:.3f}  cost {r['cost']}")

# Record the size of the final tree (depth 6, 15 leaves).
tree = joblib.load(cm.MODEL_DIR / cm.MODEL_FILES[cm.DT]).named_steps["clf"]
card["tree_size"] = {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())}
card["notes"] = [
    "Scores are trained on SMOTE-balanced data, so they rank applicants well but overstate the "
    "probability of default relative to the 30% base rate. Treat them as risk scores, not calibrated PDs.",
    "The Decision Tree's cost-based threshold is degenerate (it refuses everyone on the test set), "
    "so the demo reports its 0.5 decision as the main one.",
    "Data are from Germany in the 1970s; the models are a research demonstration, not a lending tool."]
(cm.MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

# Ten real test-set applicants (five good, five bad) for the demonstration
rng = np.random.RandomState(cm.SEED)
idx = list(rng.choice(np.where(yte == 0)[0], 5, replace=False)) + list(rng.choice(np.where(yte == 1)[0], 5, replace=False))
sample = Xte.iloc[sorted(idx)].copy()
sample.insert(0, "applicant_id", [f"T{i:03d}" for i in sorted(idx)])
sample["actual_outcome"] = np.where(yte.iloc[sorted(idx)] == 1, "Bad Risk", "Good Risk")
sample.to_csv(cm.HERE / "examples" / "sample_applicants.csv", index=False)
print("Saved models and model card to", cm.MODEL_DIR)
