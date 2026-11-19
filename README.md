# Credit Risk Model Package

**Logistic Regression and Decision Tree models for credit risk prediction**
Wilfried Mweneni Nendongo (201304914), BSc Computer Science (Honours), University of Namibia
Supervisor: Mrs Ndinelao Iitumba

This package is the software that goes with the mini thesis *A Quantitative Evaluation of Logistic Regression and Decision Tree Models for Credit Risk Prediction Using Public Financial Datasets*. It contains the trained models, the experiment code that produced every table and figure in Chapter 5, a script that checks every number in the thesis against the code, and three ways to show the models working: a command-line tool, a walkthrough notebook and a web app that scores an applicant and explains each model's decision.

The saved models reproduce the thesis results exactly. On the 200 held-out test applicants, Logistic Regression reaches recall 0.767, AUC 0.809 and a misclassification cost of 105, against 0.533, 0.685 and 181 for the Decision Tree (thesis Table 5.4).

![Web app](docs/app_screenshot.png)

## What is in the folder

| Path | What it does |
|---|---|
| `credit_model.py` | Shared core: data definitions, the modelling pipeline, scoring and the two explanation methods |
| `train_and_save.py` | Trains both models exactly as in the thesis and saves them to `models/` (about one minute) |
| `predict.py` | Command-line demo: scores an applicant, gives both decisions and explains why |
| `app.py` | Web app for the live presentation (Streamlit): scores an applicant with both models and explains each decision |
| `verify_thesis.py` | Checks 188 numbers in the thesis against the code and writes `thesis_check_report.md` |
| `demo_walkthrough.ipynb` | Step-by-step notebook for the presentation, already run with all outputs |
| `docs/demo_walkthrough.html` | The same notebook as a web page, viewable without Python |
| `models/` | The two fitted pipelines (`.joblib`) and `model_card.json` with settings and test results |
| `examples/sample_applicants.csv` | Ten real test-set applicants (five good, five bad) with their actual outcome |
| `examples/applicant.json` | A made-up applicant in the input format, to copy and edit |
| `examples/invalid_applicants.csv` | The ten sample applicants with one deliberate error, to show input validation |
| `experiments/credit_risk_experiment.py` | The main experiment behind Chapter 5 (tuning, test results, significance tests, thresholds, figures) |
| `experiments/supplementary_analysis.py` | Effect sizes, bootstrap intervals and the sixteen-variant robustness analysis (Sections 5.6 and 5.8) |
| `experiments/outputs/` | The results and figures the two scripts produced, so the checks work without the 17-minute rerun |
| `data/german.data` | Statlog German Credit data (Hofmann, 1994), 1,000 applicants, CC BY 4.0 licence |
| `PRESENTATION_GUIDE.md` | A suggested running order for the demonstration, with likely questions and answers |

## Setting up (once)

You need Python 3.11 or newer. From inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The models in `models/` were saved with scikit-learn 1.9.1. If you install a different version and loading fails, simply retrain them:

```bash
python train_and_save.py
```

This prints the selected settings and the test results, which should match the thesis.

## Demonstrating the models

### 1. Command line

```bash
python predict.py --list                       # the ten sample applicants
python predict.py --sample T115                # score and explain one of them
python predict.py --sample T115 --what-if checking_status="no checking account"
python predict.py --json examples/applicant.json
python predict.py --interactive                # type in a new applicant
python predict.py --batch examples/sample_applicants.csv --out scored.csv
python predict.py --batch examples/invalid_applicants.csv      # shows the validation message
```

For each applicant the tool prints:

* **a risk score** from each model (the estimated probability of being a Bad Risk);
* **two decisions**: at the usual 0.5 threshold, and at a cost-based threshold chosen on the training data so that approving a defaulter costs five times as much as refusing a good applicant (0.29 for Logistic Regression);
* **why Logistic Regression decided as it did**: each attribute's contribution to the log-odds of default. These contributions add up exactly to the model's output, so the explanation is the model itself rather than an approximation of it;
* **why the Decision Tree decided as it did**: the chain of yes/no rules the applicant followed, in the original units (months, Deutsche Mark), ending in a leaf.

Categorical values can be typed as readable labels (`no checking account`) or as the original UCI codes (`A14`).

### 2. Web app

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The app has a single page. Pick one of the ten sample applicants in the sidebar, or change any field, and it shows:

* each model's **risk score** and **decision** (Logistic Regression at its cost-based threshold of 0.29, the Decision Tree at 0.5);
* **why Logistic Regression decided as it did**: the attributes raising and lowering the risk most, and a bar chart of every attribute's contribution to the log-odds;
* **why the Decision Tree decided as it did**: the yes/no questions the applicant answered and the share of Bad Risk among the training applicants in the leaf they reached.

At the bottom of the page, *Score a file of applicants* takes a CSV in the format of `examples/sample_applicants.csv`, scores every row with both models and offers the result as a download.

The tables and figures of Chapter 5 are in the thesis itself; `verify_thesis.py` checks them against the code.

### Input validation and error handling

Every applicant is checked before the models see it, in the app and on the command line alike (`credit_model.to_frame`):

* all 20 attributes must be present;
* categorical values must be one of the documented labels, or the matching UCI code (`A14`), and an unknown value is refused with the list of allowed ones;
* numeric values must be numbers within the limits in `credit_model.LIMITS` (for example, loan duration 1 to 120 months). The app's form will not accept values outside them;
* a value that is allowed but lies outside the range seen in the 1,000 training records (`credit_model.TRAIN_RANGE`, for example a 90-month loan) is scored with a warning that the score is an extrapolation.

A file with a bad row is not scored, and the message names the attribute and the row; try `examples/invalid_applicants.csv`, whose third row has the duration `twelve`. If the saved models are missing or were saved with an incompatible scikit-learn version, the app and the command line say so and point to `python train_and_save.py` instead of crashing.

### 3. Notebook

```bash
jupyter notebook demo_walkthrough.ipynb
```

The notebook tells the story in order: the data and its imbalance, what is inside a saved model, the test-set results and ROC curves, one applicant explained by both models, a what-if change, batch scoring and the tree itself. It is saved with its outputs, so it can also be shown without running it, or opened as `docs/demo_walkthrough.html` in any browser.

## Checking every number in the thesis

```bash
python verify_thesis.py
```

This takes a few seconds. It re-scores the saved models on the 200 test applicants, recomputes the SMOTE finding of Section 5.8 from scratch (320 of 320 synthetic records with fractional categories, 80 and 9 of them in two leaves), and compares every other number in Chapters 5 and 6 and Appendix A with the experiment outputs. It currently reports **188 of 188 claims reproduced**, and writes the full table to `thesis_check_report.md`.

## Reproducing the full thesis results

The two experiment scripts regenerate everything in Chapter 5. Run them from the `experiments` folder:

```bash
cd experiments
python credit_risk_experiment.py ../data/german.data outputs     # about 2 minutes
python supplementary_analysis.py ../data/german.data outputs     # about 15 minutes
```

They write `results.json`, `supplementary.json` and all the figures to `experiments/outputs/`, replacing the copies shipped with the package. Run `python verify_thesis.py` afterwards to confirm the regenerated numbers still match the thesis. Every random step uses the seed 42, so the numbers match the thesis exactly with the library versions in `requirements.txt`.

## How the models were built

1. The 1,000 applicants were split 80/20, stratified so that both parts keep the 30% default rate (seed 42). The 200 test applicants were not touched until the final evaluation.
2. Each model is one pipeline: standardise the seven numeric attributes, one-hot encode the thirteen categorical ones (48 inputs), rebalance the classes with SMOTE, then fit the classifier. SMOTE is used only while fitting; it is skipped automatically when the pipeline makes predictions.
3. Settings were chosen by grid search with five-fold cross-validation, maximising the F1-score for the Bad Risk class: 5 settings for Logistic Regression (selected C = 0.1) and 896 for the Decision Tree (selected tree: depth 6, 15 leaves).
4. The cost-based threshold was chosen on out-of-fold predictions from the training data only.

## Limitations

* **Scores are not calibrated probabilities.** Because the models were trained on SMOTE-balanced data, their scores rank applicants well but overstate the chance of default compared with the real 30% rate.
* **The tree's cost-based threshold is not useful.** At 0.09 it refuses almost every applicant, so its 0.5 decision is the one to read.
* **The data are old and foreign.** They describe German borrowers in the 1970s. The package demonstrates a method and a comparison; it is not a lending tool for Namibia.
* **Protected attributes.** The data include sex, marital status and foreign-worker status. Chapter 4 of the thesis explains why these would need to be removed or tested for fairness before any real use.

## Reference

Hofmann, H. (1994). *Statlog (German credit data)* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
