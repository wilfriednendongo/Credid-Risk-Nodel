"""
Usage
  python predict.py --list                     show the ten sample applicants
  python predict.py --sample T115              score one sample applicant and explain it
  python predict.py --json examples/applicant.json
  python predict.py --interactive              type in a new applicant, question by question
  python predict.py --batch examples/sample_applicants.csv --out scored.csv
  Add --what-if checking_status="no checking account" to change one attribute and compare.

Run train_and_save.py once first so that the models/ folder exists.
"""
import argparse, json, sys, warnings
# Hide library warnings so the output stays readable in front of the panel.
warnings.filterwarnings("ignore")
import pandas as pd
import credit_model as cm

# The ten sample applicants, and the line width used for the printed report.
SAMPLES = cm.HERE / "examples" / "sample_applicants.csv"
W = 78


# Prints a separator line.
def line(ch="-"):
    print(ch * W)


# Prints one applicant's details, both models' scores and decisions, and both explanations.
def report(models, card, applicant, title="Applicant"):
    # Scoring also validates the applicant (credit_model.to_frame).
    res = cm.score(models, card, applicant)
    for w in cm.range_warnings(applicant):
        print("  WARNING: " + w)
    line("=")
    print(title)
    line("=")
    x = cm.to_frame(applicant).iloc[0]
    for c in cm.FEATURES:
        print(f"  {cm.PRETTY[c]:<48s} {x[c]}")
    line()
    print(f"  {'Model':<22s}{'Risk score':>11s}   {'Decision at 0.5':<18s}{'Cost-based decision':<22s}")
    for n, r in res.items():
        cost = f"{r['decision_cost']} (t={r['threshold_cost']:.2f})"
        print(f"  {n:<22s}{r['p_bad']:>10.1%}    {r['decision_default']:<18s}{cost:<22s}")
    print("  Risk score = estimated probability of Bad Risk. Cost-based threshold chosen on training data")
    print("  so that approving a defaulter costs five times as much as refusing a good applicant.")
    print(f"  The tree's cost-based threshold ({card['thresholds'][cm.DT]:.2f}) refuses almost everyone, so read its 0.5 decision.")

    # Logistic Regression: the 8 biggest contributions as a text bar chart, then the rest combined.
    table, intercept, logit = cm.explain_lr(models[cm.LR], applicant)
    line()
    print("WHY: Logistic Regression (contribution of each attribute to the log-odds of default)")
    for _, r in table.head(8).iterrows():
        bar = ("+" if r.contribution > 0 else "-") * min(int(abs(r.contribution) * 20) + 1, 20)
        print(f"  {cm.PRETTY[r.attribute][:32]:<32s} {str(r.value)[:20]:<20s} {r.contribution:+.2f} {bar}")
    rest = table.iloc[8:]["contribution"].sum()
    print(f"  {'(other ' + str(len(table) - 8) + ' attributes combined)':<53s} {rest:+.2f}")
    print(f"  Intercept {intercept:+.2f}  ->  log-odds {logit:+.2f}  ->  risk score {1 / (1 + 2.718281828 ** -logit):.1%}")

    # Decision Tree: the numbered yes/no questions down to the leaf.
    rules, leaf = cm.explain_dt(models[cm.DT], applicant)
    line()
    print("WHY: Decision Tree (the path this applicant followed)")
    for i, r in enumerate(rules, 1):
        print(f"  {i}. {r}")
    print(f"  -> leaf {leaf['node']}: {leaf['training_records']} training records (incl. SMOTE records), "
          f"{leaf['p_bad']:.0%} of them Bad Risk")
    line("=")
    return res


# --interactive: asks for each attribute in turn and re-asks until the answer is valid.
def ask():
    a = {}
    print("Enter the applicant's details (press Enter to accept the value in brackets).")
    defaults = pd.read_csv(SAMPLES).iloc[0].to_dict()
    for c in cm.FEATURES:
        if c in cm.CAT:
            opts = cm.options(c)
            print(f"\n{cm.PRETTY[c]}:")
            for i, o in enumerate(opts, 1):
                print(f"   {i:>2}. {o}")
            while True:
                v = input(f"  choose 1-{len(opts)} [{defaults[c]}]: ").strip()
                if not v:
                    a[c] = defaults[c]; break
                if v.isdigit() and 1 <= int(v) <= len(opts):
                    a[c] = opts[int(v) - 1]; break
                print(f"  Please type a number from 1 to {len(opts)}.")
        else:
            lo, hi = cm.LIMITS[c]
            while True:
                v = input(f"{cm.PRETTY[c]} [{defaults[c]}]: ").strip()
                if not v:
                    a[c] = defaults[c]; break
                try:
                    x = float(v)
                except ValueError:
                    print("  Please type a number."); continue
                if lo <= x <= hi:
                    a[c] = x; break
                print(f"  Please type a value between {lo:,} and {hi:,}.")
    return a


# Entry point: any expected error becomes a one-line 'Error: ...' message instead of a crash.
def main():
    try:
        run()
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        sys.exit(f"Error: {msg}")
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")


# Reads the command-line options and runs the chosen mode.
def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Exactly one mode must be given: --list, --sample, --json, --interactive or --batch.
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--sample")
    g.add_argument("--json")
    g.add_argument("--interactive", action="store_true")
    g.add_argument("--batch")
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--what-if", action="append", default=[], metavar='ATTR="VALUE"')
    args = ap.parse_args()

    models, card = cm.load_models()
    if args.list:
        s = pd.read_csv(SAMPLES)
        print(s[["applicant_id", "checking_status", "duration", "credit_amount", "purpose", "age", "actual_outcome"]].to_string(index=False))
        return
    # --batch: validate and score a whole CSV file, then save it with the scores added.
    if args.batch:
        df = pd.read_csv(args.batch)
        X = cm.to_frame(df)
        for n, m in models.items():
            p = m.predict_proba(X)[:, 1]
            key = "lr" if n == cm.LR else "dt"
            df[f"{key}_risk_score"] = p.round(4)
            t = card["thresholds"][n] if n == cm.LR else 0.5
            df[f"{key}_decision"] = ["REFER / DECLINE" if v >= t else "APPROVE" for v in p]
        df.to_csv(args.out, index=False)
        cols = [c for c in ["applicant_id", "actual_outcome"] if c in df] + ["lr_risk_score", "lr_decision", "dt_risk_score", "dt_decision"]
        print(df[cols].to_string(index=False))
        print(f"\nScored {len(df)} applicants -> {args.out}  (LR at its cost-based threshold "
              f"{card['thresholds'][cm.LR]:.2f}; DT at 0.5)")
        return
    # Pick the applicant to score: a sample, a JSON file, or typed in.
    if args.sample:
        s = pd.read_csv(SAMPLES).set_index("applicant_id")
        if args.sample not in s.index:
            sys.exit(f"Unknown sample {args.sample}. Use --list to see the ids.")
        row = s.loc[args.sample]
        applicant, title = row.drop("actual_outcome").to_dict(), f"Sample applicant {args.sample} (actual outcome: {row.actual_outcome})"
    elif args.json:
        applicant, title = json.load(open(args.json)), f"Applicant from {args.json}"
    else:
        applicant, title = ask(), "New applicant"
    base = report(models, card, applicant, title)

    # --what-if: change one or more attributes, score again and print old -> new risk.
    if args.what_if:
        changed = dict(applicant)
        for w in args.what_if:
            if "=" not in w:
                raise ValueError(f'--what-if needs ATTR="VALUE", got {w!r}')
            k, v = w.split("=", 1)
            if k.strip() not in cm.FEATURES:
                raise ValueError(f"Unknown attribute {k.strip()!r}. Valid names: {', '.join(cm.FEATURES)}")
            changed[k.strip()] = v.strip().strip('"')
        new = report(models, card, changed, "WHAT IF: " + "; ".join(args.what_if))
        for n in models:
            print(f"  {n:<22s} {base[n]['p_bad']:.1%} -> {new[n]['p_bad']:.1%}")


if __name__ == "__main__":
    main()
