"""
Shared core of the credit risk model package.

Holds the data definitions, the preprocessing and modelling pipeline, and the
explanation helpers used by train_and_save.py, predict.py, app.py and the
walkthrough notebook. The pipeline is identical to the one in
experiments/credit_risk_experiment.py, which produced the results in the thesis.

Author : Wilfried Mweneni Nendongo (201304914), University of Namibia
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ----------------------------------------------------------------------------- settings
# Fixed random seed: every split, SMOTE run and tree uses 42, so results are the same on every run.
SEED = 42
# Folder paths are built from this file's own location, so the package works from any folder.
HERE = Path(__file__).resolve().parent
# The raw Statlog German Credit file (Hofmann, 1994): 1,000 applicants, one per line.
DATA_FILE = HERE / "data" / "german.data"
# Where train_and_save.py stores the fitted models and where the app and CLI load them from.
MODEL_DIR = HERE / "models"
# Short names for the two models, used as dictionary keys everywhere.
LR, DT = "Logistic Regression", "Decision Tree"
MODEL_FILES = {LR: "logistic_regression.joblib", DT: "decision_tree.joblib"}
COST_FN, COST_FP = 5, 1          # approving a defaulter costs five times a wrongly refused good applicant

# ----------------------------------------------------------------------------- data definitions
# The 21 columns of german.data, in file order (the file has no header row).
# The last column, target, is the outcome: 1 = good payer, 2 = defaulter (Bad Risk).
COLS = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings",
        "employment", "installment_rate", "personal_status", "other_debtors", "residence_since",
        "property", "age", "other_installment_plans", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "target"]
# The 7 numeric attributes. These are standardized (scaled) before modelling.
NUM = ["duration", "credit_amount", "installment_rate", "residence_since", "age",
       "existing_credits", "num_dependents"]
# The other 13 attributes are categorical. These are one-hot encoded before modelling.
CAT = [c for c in COLS if c not in NUM + ["target"]]
# The 20 inputs a model needs: every column except the target.
FEATURES = COLS[:-1]

# german.data stores categories as codes such as A11 or A14. This dictionary decodes each
# code into a readable label, e.g. A14 -> 'no checking account'. Used when loading the data
# and when a user types a code instead of a label.
LABELS = {  # attribute codes as documented on the UCI repository
 "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM / salary", "A14": "no checking account",
 "A30": "no credits / all paid", "A31": "all paid at this bank", "A32": "existing paid duly",
 "A33": "past delays", "A34": "critical account",
 "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/TV",
 "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
 "A48": "retraining", "A49": "business", "A410": "others",
 "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown/none",
 "A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">= 7 yrs",
 "A91": "male div/sep", "A92": "female div/sep/mar", "A93": "male single", "A94": "male mar/wid",
 "A95": "female single",
 "A101": "none", "A102": "co-applicant", "A103": "guarantor",
 "A121": "real estate", "A122": "savings/life insurance", "A123": "car/other", "A124": "unknown/none",
 "A141": "bank", "A142": "stores", "A143": "none",
 "A151": "rent", "A152": "own", "A153": "for free",
 "A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled", "A174": "management/self-emp",
 "A191": "none", "A192": "yes", "A201": "yes", "A202": "no"}

# Readable names and the allowed values of every input, used for validation, the CLI and the app
PRETTY = {
 "checking_status": "Checking account balance", "duration": "Loan duration (months)",
 "credit_history": "Credit history", "purpose": "Loan purpose", "credit_amount": "Credit amount (DM)",
 "savings": "Savings balance", "employment": "Time in current employment",
 "installment_rate": "Instalment rate (% of disposable income, 1-4)",
 "personal_status": "Personal status and sex", "other_debtors": "Other debtors / guarantors",
 "residence_since": "Years at current residence (1-4)", "property": "Most valuable property",
 "age": "Age (years)", "other_installment_plans": "Other instalment plans", "housing": "Housing",
 "existing_credits": "Existing credits at this bank", "job": "Job",
 "num_dependents": "Number of dependants", "telephone": "Registered telephone",
 "foreign_worker": "Foreign worker"}
# Each categorical attribute's codes share a prefix (checking_status uses A11 to A14, so A1).
PREFIX = {"checking_status": "A1", "credit_history": "A3", "purpose": "A4", "savings": "A6",
          "employment": "A7", "personal_status": "A9", "other_debtors": "A10", "property": "A12",
          "other_installment_plans": "A14", "housing": "A15", "job": "A17", "telephone": "A19",
          "foreign_worker": "A20"}


# Numeric inputs: hard limits (values outside are rejected as invalid) and the range seen in the
# 1,000 training records (values outside are accepted, but the score is an extrapolation).
LIMITS = {"duration": (1, 120), "credit_amount": (100, 100_000), "installment_rate": (1, 4),
          "residence_since": (1, 4), "age": (18, 100), "existing_credits": (1, 10), "num_dependents": (0, 10)}
TRAIN_RANGE = {"duration": (4, 72), "credit_amount": (250, 18_424), "installment_rate": (1, 4),
               "residence_since": (1, 4), "age": (19, 75), "existing_credits": (1, 4), "num_dependents": (1, 2)}


# Builds the list of allowed labels for one categorical attribute (used for dropdowns and validation).
def options(col):
    """Allowed labels of a categorical attribute, in code order."""
    p = PREFIX[col]
    # Keep codes that are the prefix plus one character, e.g. A11..A14 for prefix A1.
    codes = [c for c in LABELS if c.startswith(p) and len(c) == len(p) + 1]
    # A410 ('others') has two extra characters, so it is added by hand.
    if col == "purpose":
        codes.append("A410")
    return [LABELS[c] for c in codes]


# ----------------------------------------------------------------------------- loading the data
# THIS IS WHERE german.data IS READ AND DECODED.
def load_data(path=DATA_FILE):
    """Load german.data (space or comma separated), decode labels, return X and y (1 = Bad Risk)."""
    # Read the file. It has no header, so the column names come from COLS.
    # sep=None lets pandas detect whether values are separated by spaces or commas.
    raw = pd.read_csv(path, header=None, names=COLS, sep=None, engine="python")
    # Copy the raw table, then decode every categorical column from codes (A11, A14 ...) to
    # readable labels with LABELS, so the models and the app work with words, not codes.
    df = raw.copy()
    for c in CAT:
        df[c] = df[c].astype(str).map(LABELS)
    # Recode the outcome: 2 (defaulted) becomes 1 = Bad Risk, 1 (repaid) becomes 0 = Good Risk.
    df["target"] = (raw["target"] == 2).astype(int)
    # X = the 20 inputs, y = the outcome.
    return df.drop(columns="target"), df["target"]


# ----------------------------------------------------------------------------- input validation
# Every applicant passes through this function before a model sees it (app, CLI, file upload).
def to_frame(applicants):
    """Turn one applicant (dict) or many (list of dicts / DataFrame) into a validated model input.
    Categorical values may be given as labels (e.g. 'no checking account') or UCI codes (e.g. 'A14')."""
    # Accept one applicant (a dict) or many, and turn them into a table.
    if isinstance(applicants, dict):
        applicants = [applicants]
    df = pd.DataFrame(applicants).copy().reset_index(drop=True)
    # Check 1: all 20 attributes must be present.
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing attributes: " + ", ".join(missing))
    for c in CAT:
        # Check 2: categories. Codes like A14 are translated to labels; labels stay as they are.
        df[c] = df[c].astype(str).str.strip().map(lambda v: LABELS.get(v, v))
        # Any value that is not an allowed label is refused, and the message lists the allowed ones.
        bad = sorted(set(df[c]) - set(options(c)))
        if bad:
            raise ValueError(f"Unknown value(s) {bad} for '{PRETTY[c]}'. Allowed: {options(c)}")
    if df.empty:
        raise ValueError("No applicants to score.")
    for c in NUM:
        # Check 3: numbers. Text such as 'twelve' becomes NaN (missing) here, and is refused below.
        values = pd.to_numeric(df[c], errors="coerce")
        if values.isna().any():
            rows = [int(i) + 1 for i in values[values.isna()].index[:5]]
            raise ValueError(f"'{PRETTY[c]}' must be a number" + (f" (problem in row(s) {', '.join(map(str, rows))})." if len(df) > 1 else "."))
        # Check 4: the number must lie within the hard limits in LIMITS.
        lo, hi = LIMITS[c]
        out = values[(values < lo) | (values > hi)]
        if len(out):
            raise ValueError(f"'{PRETTY[c]}' must be between {lo:,} and {hi:,}; got {out.iloc[0]:g}"
                             + (f" (row {int(out.index[0]) + 1})." if len(df) > 1 else "."))
        df[c] = values
    # Return the clean table, columns in the order the models expect.
    return df[FEATURES]


# Not an error: flags values the model never saw in training, so the user knows to be careful.
def range_warnings(applicant):
    """Valid inputs that lie outside the range seen in training: the model can score them,
    but it is extrapolating, so the score is less reliable."""
    msgs = []
    for c, (lo, hi) in TRAIN_RANGE.items():
        v = float(applicant[c])
        if v < lo or v > hi:
            msgs.append(f"{PRETTY[c]} = {v:g} is outside the training range ({lo:,} to {hi:,}); "
                        "the score is an extrapolation.")
    return msgs


# ----------------------------------------------------------------------------- pipeline
# ----------------------------------------------------------------------------- pipeline
# Step 1 of the pipeline: scale numeric columns to mean 0 and standard deviation 1, and turn each
# categorical column into 0/1 dummy columns (drop='first' drops one reference category).
def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT)])


# The full pipeline: preprocessing -> SMOTE (balances the classes) -> classifier.
# imblearn's Pipeline runs SMOTE only during fit(), never during predict().
def pipe(clf):
    """SMOTE sits inside the pipeline, so it is used only when fitting and is skipped at prediction."""
    return Pipeline([("pre", preprocessor()), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])


# The settings tried in the grid search: 5 values of C for Logistic Regression (smaller C =
# stronger L2 penalty), and 7 x 4 x 4 x 2 x 4 = 896 combinations for the Decision Tree.
GRIDS = {
 LR: (LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
      {"clf__C": [0.01, 0.1, 1, 10, 100]}),
 DT: (DecisionTreeClassifier(random_state=SEED),
      {"clf__max_depth": [3, 4, 5, 6, 8, 10, None],
       "clf__min_samples_split": [2, 5, 10, 20],
       "clf__min_samples_leaf": [1, 5, 10, 20],
       "clf__criterion": ["gini", "entropy"],
       "clf__ccp_alpha": [0.0, 0.001, 0.005, 0.01]})}


# Loads the two fitted pipelines (.joblib files) and the model card (settings, thresholds, results).
def load_models(model_dir=MODEL_DIR):
    """Return ({name: fitted pipeline}, model card dict)."""
    model_dir = Path(model_dir)
    if not (model_dir / "model_card.json").exists():
        raise FileNotFoundError("No saved models found. Run:  python train_and_save.py")
    # joblib.load turns each saved file back into the fitted Python pipeline object.
    models = {n: joblib.load(model_dir / f) for n, f in MODEL_FILES.items()}
    card = json.loads((model_dir / "model_card.json").read_text())
    return models, card


# ----------------------------------------------------------------------------- explanations
# ----------------------------------------------------------------------------- explanations
# Names of the 48 columns after encoding, e.g. 'checking_status_no checking account'.
def _feature_names(model):
    return [f.split("__", 1)[1] for f in model.named_steps["pre"].get_feature_names_out()]


# Maps an encoded column back to its original attribute (the longest matching name wins).
def _attribute_of(feature):
    if feature in NUM:
        return feature
    return next(c for c in sorted(CAT, key=len, reverse=True) if feature.startswith(c + "_"))


def explain_lr(model, applicant):
    """Split the Logistic Regression log-odds of one applicant into one contribution per attribute.

    Each contribution is coefficient x encoded value, summed over the dummy columns of an attribute.
    It is measured against a reference applicant with average numeric values and the first
    (dropped) category of every categorical attribute. Contributions plus the intercept give the
    log-odds exactly, so the explanation is faithful to the model rather than an approximation."""
    x = to_frame(applicant)
    # Encode the applicant exactly as the model sees it.
    z = model.named_steps["pre"].transform(x)[0]
    clf = model.named_steps["clf"]
    # Contribution of each encoded column = its coefficient x its value.
    contrib = clf.coef_[0] * z
    per_attr = {}
    # Add up the dummy columns of each categorical attribute into one number per attribute.
    for f, c in zip(_feature_names(model), contrib):
        a = _attribute_of(f)
        per_attr[a] = per_attr.get(a, 0.0) + float(c)
    table = pd.DataFrame({"attribute": list(per_attr), "value": [x.iloc[0][a] for a in per_attr],
                          "contribution": list(per_attr.values())})
    table["direction"] = np.where(table["contribution"] > 0, "raises risk", "lowers risk")
    # Sort so the attributes with the biggest effect come first.
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    intercept = float(clf.intercept_[0])
    # Intercept + all contributions = the model's log-odds; the sigmoid of this is the risk score.
    logit = intercept + table["contribution"].sum()
    return table, intercept, float(logit)


def explain_dt(model, applicant):
    """Return the chain of rules the Decision Tree applied to one applicant, in original units."""
    x = to_frame(applicant)
    pre = model.named_steps["pre"]
    z = pre.transform(x)
    tree = model.named_steps["clf"].tree_
    names = _feature_names(model)
    scaler = pre.named_transformers_["num"]
    # The list of tree nodes this applicant passes through, from the root to a leaf.
    path = model.named_steps["clf"].decision_path(z).indices
    rules = []
    for node in path:
        if tree.children_left[node] == tree.children_right[node]:      # leaf
            continue
        # f = the column tested at this node, t = the split value; values <= t go left.
        f, t = tree.feature[node], tree.threshold[node]
        name, goes_left = names[f], z[0, f] <= t
        if name in NUM:
            i = NUM.index(name)
            # The tree split on scaled values, so undo the scaling to show months or DM.
            cut = t * scaler.scale_[i] + scaler.mean_[i]
            rule = f"{PRETTY[name]} {'<=' if goes_left else '>'} {cut:,.1f}   (applicant: {x.iloc[0][name]:,})"
        else:
            a = _attribute_of(name)
            # Dummy column: the question is whether the applicant has this category.
            level = name[len(a) + 1:]
            rule = f"{PRETTY[a]} {'is not' if goes_left else 'is'} '{level}'   (applicant: {x.iloc[0][a]})"
        if not rules or rules[-1] != rule:   # SMOTE's fractional dummies can make the tree test one category twice
            rules.append(rule)
    # The last node is the leaf; its share of Bad Risk records is the tree's risk score.
    leaf = path[-1]
    p_bad = float(model.predict_proba(x)[0, 1])
    return rules, {"node": int(leaf), "training_records": int(tree.n_node_samples[leaf]), "p_bad": p_bad}


# Used by predict.py: the risk score of each model and its decision at two thresholds.
def score(models, card, applicant):
    """Probability of Bad Risk and decisions at the default (0.5) and cost-based thresholds."""
    x = to_frame(applicant)
    out = {}
    for n, m in models.items():
        # predict_proba gives [P(Good), P(Bad)]; we keep P(Bad) as the risk score.
        p = float(m.predict_proba(x)[0, 1])
        # Cost-based threshold chosen in train_and_save.py (0.29 for LR, 0.09 for the tree).
        t = card["thresholds"][n]
        out[n] = {"p_bad": p,
                  "decision_default": "REFER / DECLINE" if p >= 0.5 else "APPROVE",
                  "threshold_cost": t,
                  "decision_cost": "REFER / DECLINE" if p >= t else "APPROVE"}
    return out
