"""
Usage : streamlit run app.py      (then open the address it prints, usually http://localhost:8501)
Needs : the saved models in models/ (run train_and_save.py once if they will not load)

Scores one applicant with both models and shows, for each model, the risk score, the decision
and the reasons behind it. A file of applicants can also be scored at the bottom of the page.
The full results of Chapter 5 are in the thesis; verify_thesis.py checks them against the code.

Data handling: inputs are validated twice. The form only accepts values inside the limits in
credit_model.LIMITS, and credit_model.to_frame() checks every applicant again (allowed category
labels or UCI codes, numbers, limits) before the models see it. Values that are valid but outside
the range seen in training are scored with a warning.
"""
import math
import altair as alt
import pandas as pd
import streamlit as st
import credit_model as cm

# Page title and wide layout, so the two models can sit side by side.
st.set_page_config(page_title="Credit risk demo", layout="wide")
LR, DT = cm.LR, cm.DT


# Load the saved models once and keep them in memory (not on every click).
@st.cache_resource
def models():
    return cm.load_models()


# Small helper that joins reasons into a readable sentence.
def listing(rows):
    """'a', 'a and b', 'a, b and c'."""
    rows = list(rows)
    return rows[0] if len(rows) == 1 else ", ".join(rows[:-1]) + " and " + rows[-1]


# Turns a risk score into a decision. LIVE CHANGE: edit the threshold line below.
def decide(name, p, card):
    """LR decides at its cost-based threshold, the tree at 0.5 (its cost-based threshold refuses everyone)."""
    t = card["thresholds"][name] if name == LR else 0.5
    return ("REFER / DECLINE" if p >= t else "APPROVE"), t


# ----------------------------------------------------------------------------- page heading
st.title("Credit risk prediction: Logistic Regression vs Decision Tree")
st.caption("Models trained on the Statlog German Credit data (Hofmann, 1994).")

# ----------------------------------------------------------------------------- load models (fail clearly)
try:
    # If the model files are missing or incompatible, show a clear message and stop (no crash).
    MODELS, CARD = models()
except Exception as e:  # missing files, or files saved with an incompatible scikit-learn version
    st.error(f"The saved models could not be loaded ({e}). In a terminal in this folder run "
             "`python train_and_save.py` (about one minute), then reload this page.")
    st.stop()

try:
    # The ten sample applicants for the sidebar dropdown (optional: the app still works without them).
    SAMPLES = pd.read_csv(cm.HERE / "examples" / "sample_applicants.csv").set_index("applicant_id")
except Exception:
    SAMPLES = None

# ----------------------------------------------------------------------------- applicant form
with st.sidebar:
    st.header("Applicant")
    if SAMPLES is not None:
        # Choose which sample applicant fills the form.
        pick = st.selectbox("Start from a test-set applicant", list(SAMPLES.index),
                            format_func=lambda i: f"{i} (actual: {SAMPLES.loc[i, 'actual_outcome']})")
        base = SAMPLES.loc[pick]
    else:
        st.warning("examples/sample_applicants.csv is missing, so the form starts from neutral values.")
        pick, base = "new", {c: (cm.options(c)[0] if c in cm.CAT else cm.TRAIN_RANGE[c][0]) for c in cm.FEATURES}
    st.caption("Change any field below to see how each model responds.")
    applicant = {}
    for c in cm.FEATURES:
        if c in cm.CAT:
            opts = cm.options(c)
            start = opts.index(base[c]) if base[c] in opts else 0
            # Categories use dropdowns, so the user can only pick an allowed value.
            applicant[c] = st.selectbox(cm.PRETTY[c], opts, index=start, key=f"{pick}-{c}")
        else:
            lo, hi = cm.LIMITS[c]
            start = min(max(int(base[c]), lo), hi)
            # Numbers use number boxes limited to LIMITS (first layer of validation);
            # the help text shows the allowed range and the range seen in training.
            applicant[c] = st.number_input(cm.PRETTY[c], min_value=lo, max_value=hi, value=start, step=1,
                                           key=f"{pick}-{c}", help=f"Allowed {lo:,} to {hi:,}. "
                                           f"Seen in training: {cm.TRAIN_RANGE[c][0]:,} to {cm.TRAIN_RANGE[c][1]:,}.")

# ----------------------------------------------------------------------------- scores and decisions
try:
    # Score with both models and build both explanations. to_frame() inside these calls
    # validates the applicant again (second layer); any problem is shown as an error message.
    res = cm.score(MODELS, CARD, applicant)
    lr_tab, intercept, logit = cm.explain_lr(MODELS[LR], applicant)
    rules, leaf = cm.explain_dt(MODELS[DT], applicant)
except ValueError as e:
    st.error(f"This applicant cannot be scored: {e}")
    st.stop()

# Yellow warning for values outside the training range (scored, but less reliable).
for w in cm.range_warnings(applicant):
    st.warning(w)

decisions = {}
cols = st.columns(2)
# ----------------------------------------------------------------------------- the two result cards
# Same layout for both models: risk score, coloured decision, progress bar and the threshold used.
for col, n in zip(cols, (LR, DT)):
    p = res[n]["p_bad"]
    decision, t = decide(n, p, CARD)
    decisions[n] = decision
    rule = f"cost-based threshold {t:.2f}" if n == LR else "threshold 0.50"
    with col:
        st.subheader(n)
        st.metric("Risk of default", f"{p:.1%}")
        # Green for APPROVE, red for REFER / DECLINE. LIVE CHANGE: swap the colours here.
        st.markdown(f"Decision: **:{'red' if decision != 'APPROVE' else 'green'}[{decision}]**")
        st.progress(min(p, 1.0))
        st.caption(f"Declines when the risk score is above the {rule}.")

if SAMPLES is not None:
    st.info(f"Actual outcome of {pick} before any edits: **{base['actual_outcome']}**.")

# ----------------------------------------------------------------------------- reasons
left, right = st.columns(2)
# Left column: Logistic Regression's reasons (contribution of each attribute).
with left:
    st.subheader("Why: Logistic Regression")
    t = lr_tab.copy()
    t["attribute"] = t["attribute"].map(cm.PRETTY)
    t["value"] = t["value"].astype(str)
    # Top 3 attributes raising and lowering the risk. LIVE CHANGE: [:3] -> [:5] for five reasons.
    up = [f"{a}: {v}" for a, v, c in zip(t["attribute"], t["value"], t["contribution"]) if c > 0][:3]
    down = [f"{a}: {v}" for a, v, c in zip(t["attribute"], t["value"], t["contribution"]) if c < 0][:3]
    verb = "approves" if decisions[LR] == "APPROVE" else "refers or declines"
    st.markdown(f"The model **{verb}** this applicant. It weighs every attribute at once:")
    if up:
        st.markdown(f"- **Raising the risk most:** {listing(up)}")
    if down:
        st.markdown(f"- **Lowering the risk most:** {listing(down)}")
    # The 10 biggest contributions go in the bar chart. LIVE CHANGE: 10 -> 20 shows all attributes.
    top = t.head(10).copy()
    top["effect"] = top["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    st.altair_chart(alt.Chart(top).mark_bar().encode(
        x=alt.X("contribution:Q", title="Contribution to log-odds of default"),
        y=alt.Y("attribute:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("effect:N", title=None,
                        scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#c0392b", "#2e86c1"])),
        tooltip=["attribute", "value", alt.Tooltip("contribution:Q", format="+.3f")]), width="stretch")
    st.caption(f"Positive bars push towards Bad Risk. Intercept {intercept:+.2f} plus all contributions "
               f"gives log-odds {logit:+.2f}, i.e. a score of {1 / (1 + math.exp(-logit)):.1%}.")

# Right column: the Decision Tree's reasons (the yes/no path to a leaf).
with right:
    st.subheader("Why: Decision Tree")
    verb = "approves" if decisions[DT] == "APPROVE" else "refers or declines"
    st.markdown(f"The model **{verb}** this applicant after answering {len(rules)} yes/no questions:")
    for i, r in enumerate(rules, 1):
        st.markdown(f"{i}. {r}")
    st.markdown(f"These answers lead to a group of **{leaf['training_records']} training applicants** "
                f"(including SMOTE records), **{leaf['p_bad']:.0%}** of whom were Bad Risk. "
                f"That share is the applicant's risk score.")
    st.caption("Attributes that are not on this path have no effect on the tree's decision.")

# ----------------------------------------------------------------------------- score a file
st.divider()
# Upload a CSV, validate every row, score it with both models and offer it for download.
with st.expander("Score a file of applicants (CSV)"):
    st.markdown("Upload a CSV with one row per applicant and the 20 attribute columns, as in "
                "`examples/sample_applicants.csv`. Categories can be labels or UCI codes (e.g. `A14`). "
                "Every row is validated before scoring.")
    up_file = st.file_uploader("CSV file", type=["csv"])
    if up_file is not None:
        try:
            df = pd.read_csv(up_file)
            # A file with any bad row is refused as a whole, and the message names the row.
            X = cm.to_frame(df)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            st.error(f"The file was not scored: {e}")
        else:
            for n, m in MODELS.items():
                key = "lr" if n == LR else "dt"
                p = m.predict_proba(X)[:, 1]
                df[f"{key}_risk_score"] = p.round(4)
                df[f"{key}_decision"] = [decide(n, v, CARD)[0] for v in p]
            n_warn = sum(bool(cm.range_warnings(r)) for r in X.to_dict("records"))
            st.success(f"Scored {len(df)} applicants." + (f" {n_warn} have values outside the training "
                       "range, so their scores are extrapolations." if n_warn else ""))
            show = [c for c in ["applicant_id", "actual_outcome"] if c in df] + \
                   ["lr_risk_score", "lr_decision", "dt_risk_score", "dt_decision"]
            st.dataframe(df[show], hide_index=True, width="stretch")
            # Data output: the scored file goes back to the user as a CSV.
            st.download_button("Download the scored file", df.to_csv(index=False), "scored.csv", "text/csv")
