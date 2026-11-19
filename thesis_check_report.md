# Thesis verification report

**188 of 188 claims reproduced.** Rows marked *(live)* were recomputed just now from the saved models; the rest come from the experiment outputs in `experiments/outputs`.

| Where in the thesis | Claim | Thesis | Code | Result |
|---|---|---|---|---|
| Table 5.4 (saved model, live) | Logistic Regression accuracy | 0.755 | 0.755 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression precision | 0.568 | 0.5679 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression recall | 0.767 | 0.7667 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression specificity | 0.75 | 0.75 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression F1 | 0.652 | 0.6525 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression AUC | 0.809 | 0.8093 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression AP | 0.664 | 0.6642 | PASS |
| Table 5.4 (saved model, live) | Logistic Regression cost | 105 | 105.0 | PASS |
| Table 5.4 (saved model, live) | Decision Tree accuracy | 0.655 | 0.655 | PASS |
| Table 5.4 (saved model, live) | Decision Tree precision | 0.438 | 0.4384 | PASS |
| Table 5.4 (saved model, live) | Decision Tree recall | 0.533 | 0.5333 | PASS |
| Table 5.4 (saved model, live) | Decision Tree specificity | 0.707 | 0.7071 | PASS |
| Table 5.4 (saved model, live) | Decision Tree F1 | 0.481 | 0.4812 | PASS |
| Table 5.4 (saved model, live) | Decision Tree AUC | 0.685 | 0.6852 | PASS |
| Table 5.4 (saved model, live) | Decision Tree AP | 0.437 | 0.4371 | PASS |
| Table 5.4 (saved model, live) | Decision Tree cost | 181 | 181.0 | PASS |
| Table 5.6 (live) | LR right, tree wrong | 31 | 31.0 | PASS |
| Table 5.6 (live) | tree right, LR wrong | 11 | 11.0 | PASS |
| Table 5.6 (live) | both right | 120 | 120.0 | PASS |
| Section 5.6.1 (live) | exact McNemar p | 0.003 | 0.0029 | PASS |
| Section 5.4 (live) | pruned tree depth | 6 | 6.0 | PASS |
| Section 5.4 (live) | pruned tree leaves | 15 | 15.0 | PASS |
| Section 5.4 (live) | inputs used by the tree | 11 | 11.0 | PASS |
| Section 5.5 (live) | tree F1 on training data | 0.644 | 0.6435 | PASS |
| Section 5.2 (live) | duplicate records | 0 | 0.0 | PASS |
| Section 5.3 (live) | synthetic defaulters added by SMOTE | 320 | 320.0 | PASS |
| Section 5.8 (live) | synthetic records with a fractional dummy | 320 | 320.0 | PASS |
| Section 5.8 (live) | records in the synthetic-only leaves | 80 + 9 | 80 + 9 | PASS |
| Section 5.5 (live) | LR recall at the tree's specificity | 0.767 | 0.7667 | PASS |
| Section 5.5 (live) | LR specificity at the tree's recall | 0.893 | 0.8929 | PASS |
| Table A.2 (live) | T115 LR risk score | 0.502 | 0.5017 | PASS |
| Table A.2 (live) | T115 tree risk score | 0.156 | 0.1558 | PASS |
| Table A.2 (live) | T115 LR decision at 0.29 | REFER / DECLINE | REFER / DECLINE | PASS |
| Table A.2 (live) | T115 tree decision at 0.5 | APPROVE | APPROVE | PASS |
| Appendix A.2 (live) | T115 LR score with no checking account | 0.183 | 0.1827 | PASS |
| Table A.2 (live) | T115 LR contribution of checking_status | 0.52 | 0.5214 | PASS |
| Table A.2 (live) | T115 LR contribution of foreign_worker | 0.48 | 0.4814 | PASS |
| Table A.2 (live) | T115 LR contribution of savings | 0.35 | 0.3521 | PASS |
| Table A.2 (live) | T115 LR contribution of other_installment_plans | -0.69 | -0.6946 | PASS |
| Table A.2 (live) | T115 LR contribution of employment | -0.58 | -0.5817 | PASS |
| Section 5.2 | records | 1000 | 1000.0 | PASS |
| Section 5.2 | missing values | 0 | 0.0 | PASS |
| Section 5.2 | defaulters | 300 | 300.0 | PASS |
| Table 5.1 | duration mean | 20.9 | 20.9 | PASS |
| Table 5.1 | duration std | 12.1 | 12.1 | PASS |
| Table 5.1 | duration min | 4 | 4.0 | PASS |
| Table 5.1 | duration 50% | 18 | 18.0 | PASS |
| Table 5.1 | duration max | 72 | 72.0 | PASS |
| Table 5.1 | credit_amount mean | 3271.3 | 3271.3 | PASS |
| Table 5.1 | credit_amount std | 2822.7 | 2822.7 | PASS |
| Table 5.1 | credit_amount min | 250 | 250.0 | PASS |
| Table 5.1 | credit_amount 50% | 2319.5 | 2319.5 | PASS |
| Table 5.1 | credit_amount max | 18424 | 18424.0 | PASS |
| Table 5.1 | age mean | 35.5 | 35.5 | PASS |
| Table 5.1 | age std | 11.4 | 11.4 | PASS |
| Table 5.1 | age min | 19 | 19.0 | PASS |
| Table 5.1 | age 50% | 33 | 33.0 | PASS |
| Table 5.1 | age max | 75 | 75.0 | PASS |
| Table 5.2 | default rate, checking account no checking account | 11.7 | 11.7 | PASS |
| Table 5.2 | default rate, checking account < 0 DM | 49.3 | 49.3 | PASS |
| Table 5.2 | default rate, checking account 0-200 DM | 39.0 | 39.0 | PASS |
| Table 5.2 | default rate, checking account >= 200 DM / salary | 22.2 | 22.2 | PASS |
| Table 5.2 | default rate, credit history critical account | 17.1 | 17.1 | PASS |
| Table 5.2 | default rate, credit history all paid at this bank | 57.1 | 57.1 | PASS |
| Table 5.2 | default rate, credit history no credits / all paid | 62.5 | 62.5 | PASS |
| Table 5.2 | default rate, housing own | 26.1 | 26.1 | PASS |
| Table 5.2 | default rate, housing rent | 39.1 | 39.1 | PASS |
| Table 5.2 | default rate, housing for free | 40.7 | 40.7 | PASS |
| Section 5.3 | encoded inputs | 48 | 48.0 | PASS |
| Section 5.3 | largest correlation |r| | 0.74 | 0.738 | PASS |
| Section 5.3 | pairs above 0.85 | 0 | 0.0 | PASS |
| Section 5.3 | records per class after SMOTE | 560 | 560.0 | PASS |
| Table 5.3 | LR selected C | 0.1 | 0.1 | PASS |
| Table 5.3 | LR CV F1 (validation) | 0.599 | 0.5988 | PASS |
| Table 5.3 | LR CV F1 (training folds) | 0.637 | 0.6373 | PASS |
| Table 5.3 | tree CV F1 (validation) | 0.574 | 0.5736 | PASS |
| Table 5.3 | tree CV F1 (training folds) | 0.678 | 0.6778 | PASS |
| Table 3.1 | tree grid size | 896 | 896.0 | PASS |
| Section 5.4 | validation F1 at depth 1 (curve peak) | 0.558 | 0.5583 | PASS |
| Table 5.4 | unpruned tree accuracy | 0.69 | 0.69 | PASS |
| Table 5.4 | unpruned tree recall | 0.55 | 0.55 | PASS |
| Table 5.4 | unpruned tree f1 | 0.516 | 0.5156 | PASS |
| Table 5.4 | unpruned tree auc | 0.65 | 0.65 | PASS |
| Table 5.4 | unpruned tree cost | 170 | 170.0 | PASS |
| Table 5.4 | majority baseline accuracy | 0.7 | 0.7 | PASS |
| Table 5.4 | majority baseline cost | 300 | 300.0 | PASS |
| Table 5.5 | Logistic Regression good risk precision | 0.882 | 0.882 | PASS |
| Table 5.5 | Logistic Regression good risk f1 | 0.811 | 0.811 | PASS |
| Table 5.5 | Decision Tree good risk precision | 0.78 | 0.78 | PASS |
| Table 5.5 | Decision Tree good risk f1 | 0.742 | 0.742 | PASS |
| Section 5.6.1 | McNemar odds ratio | 2.82 | 2.82 | PASS |
| Section 5.6.1 | odds ratio CI lower | 1.38 | 1.38 | PASS |
| Section 5.6.1 | odds ratio CI upper | 6.21 | 6.21 | PASS |
| Table 5.7 | f1: LR mean | 0.608 | 0.6079 | PASS |
| Table 5.7 | f1: tree mean | 0.501 | 0.5011 | PASS |
| Table 5.7 | f1: LR wins of 100 | 93 | 93.0 | PASS |
| Table 5.7 | f1: corrected t | 4.21 | 4.21 | PASS |
| Table 5.7 | f1: effect size d | 1.47 | 1.47 | PASS |
| Table 5.7 | f1: 5x2cv F | 8.9 | 8.9 | PASS |
| Table 5.7 | f1: 5x2cv p | 0.013 | 0.0131 | PASS |
| Table 5.7 | auc: LR mean | 0.789 | 0.7887 | PASS |
| Table 5.7 | auc: tree mean | 0.717 | 0.7168 | PASS |
| Table 5.7 | auc: LR wins of 100 | 94 | 94.0 | PASS |
| Table 5.7 | auc: corrected t | 4.77 | 4.77 | PASS |
| Table 5.7 | auc: effect size d | 1.66 | 1.66 | PASS |
| Table 5.7 | auc: 5x2cv F | 16.38 | 16.38 | PASS |
| Table 5.7 | auc: 5x2cv p | 0.003 | 0.0033 | PASS |
| Table 5.7 | acc: LR mean | 0.724 | 0.7236 | PASS |
| Table 5.7 | acc: tree mean | 0.695 | 0.6951 | PASS |
| Table 5.7 | acc: LR wins of 100 | 72 | 72.0 | PASS |
| Table 5.7 | acc: corrected t | 1.82 | 1.82 | PASS |
| Table 5.7 | acc: effect size d | 0.63 | 0.63 | PASS |
| Table 5.7 | acc: 5x2cv F | 21.95 | 21.95 | PASS |
| Table 5.7 | acc: 5x2cv p | 0.002 | 0.0016 | PASS |
| Table 5.7 | accuracy: corrected t-test p | 0.072 | 0.0719 | PASS |
| Section 5.6.2 | tree F1 SD across folds | 0.08 | 0.0804 | PASS |
| Section 5.6.2 | LR F1 SD across folds | 0.054 | 0.0538 | PASS |
| Figure 5.5 | bootstrap recall difference | 0.233 | 0.233 | PASS |
| Figure 5.5 | bootstrap recall CI lower | 0.133 | 0.133 | PASS |
| Figure 5.5 | bootstrap recall CI upper | 0.35 | 0.35 | PASS |
| Figure 5.5 | bootstrap f1 difference | 0.171 | 0.171 | PASS |
| Figure 5.5 | bootstrap f1 CI lower | 0.093 | 0.093 | PASS |
| Figure 5.5 | bootstrap f1 CI upper | 0.261 | 0.261 | PASS |
| Figure 5.5 | bootstrap auc difference | 0.124 | 0.124 | PASS |
| Figure 5.5 | bootstrap auc CI lower | 0.057 | 0.057 | PASS |
| Figure 5.5 | bootstrap auc CI upper | 0.191 | 0.191 | PASS |
| Figure 5.5 | bootstrap specificity difference | 0.043 | 0.043 | PASS |
| Figure 5.5 | bootstrap specificity CI lower | -0.029 | -0.029 | PASS |
| Figure 5.5 | bootstrap specificity CI upper | 0.114 | 0.114 | PASS |
| Figure 5.5 | bootstrap cost difference | -76 | -76.0 | PASS |
| Figure 5.5 | bootstrap cost CI lower | -110 | -110.0 | PASS |
| Figure 5.5 | bootstrap cost CI upper | -43 | -43.0 | PASS |
| Figure 5.5 | CV f1 difference | 0.107 | 0.107 | PASS |
| Figure 5.5 | CV f1 CI | 0.056 to 0.157 | 0.056 to 0.157 | PASS |
| Figure 5.5 | CV auc difference | 0.072 | 0.072 | PASS |
| Figure 5.5 | CV auc CI | 0.042 to 0.102 | 0.042 to 0.102 | PASS |
| Figure 5.5 | CV acc difference | 0.029 | 0.029 | PASS |
| Figure 5.5 | CV acc CI | -0.003 to 0.06 | -0.003 to 0.06 | PASS |
| Section 5.7 | LR cost-based threshold | 0.29 | 0.29 | PASS |
| Section 5.7 | tree cost-based threshold | 0.09 | 0.09 | PASS |
| Section 5.7 | LR out-of-fold cost at 0.5 | 513 | 513.0 | PASS |
| Section 5.7 | LR out-of-fold cost at 0.29 | 422 | 422.0 | PASS |
| Table 5.8 | LR at 0.29 recall | 0.883 | 0.8833 | PASS |
| Table 5.8 | LR at 0.29 precision | 0.414 | 0.4141 | PASS |
| Table 5.8 | LR at 0.29 specificity | 0.464 | 0.4643 | PASS |
| Table 5.8 | LR at 0.29 f1 | 0.564 | 0.5638 | PASS |
| Table 5.8 | LR at 0.29 cost | 110 | 110.0 | PASS |
| Table 5.8 | tree at 0.09 cost (refuses everyone) | 140 | 140.0 | PASS |
| Table 5.9 | 1:1 LR cost | 44 | 44.0 | PASS |
| Table 5.9 | 1:1 tree cost | 69 | 69.0 | PASS |
| Table 5.9 | 1:1 approve-all cost | 60 | 60.0 | PASS |
| Table 5.9 | 1:1 refuse-all cost | 140 | 140.0 | PASS |
| Table 5.9 | 2:1 LR cost | 69 | 69.0 | PASS |
| Table 5.9 | 2:1 tree cost | 97 | 97.0 | PASS |
| Table 5.9 | 2:1 approve-all cost | 120 | 120.0 | PASS |
| Table 5.9 | 2:1 refuse-all cost | 140 | 140.0 | PASS |
| Table 5.9 | 5:1 LR cost | 110 | 110.0 | PASS |
| Table 5.9 | 5:1 tree cost | 140 | 140.0 | PASS |
| Table 5.9 | 5:1 approve-all cost | 300 | 300.0 | PASS |
| Table 5.9 | 5:1 refuse-all cost | 140 | 140.0 | PASS |
| Table 5.9 | 10:1 LR cost | 114 | 114.0 | PASS |
| Table 5.9 | 10:1 tree cost | 140 | 140.0 | PASS |
| Table 5.9 | 10:1 approve-all cost | 600 | 600.0 | PASS |
| Table 5.9 | 10:1 refuse-all cost | 140 | 140.0 | PASS |
| Table 5.10 | tree variants | 10 | 10.0 | PASS |
| Table 5.10 | LR variants | 6 | 6.0 | PASS |
| Section 5.8 | smallest LR - tree CV AUC gap | 0.063 | 0.063 | PASS |
| Section 5.8 | largest LR - tree CV AUC gap | 0.106 | 0.106 | PASS |
| Section 5.8 | every tree variant significantly behind (max p) | 0.0 | 0.0 | PASS |
| Section 5.8 | best tree CV AUC | 0.725 | 0.725 | PASS |
| Section 5.8 | best tree | DT: class weighting, ordinal encoding | DT: class weighting, ordinal encoding | PASS |
| Section 5.8 | ordinal tree test recall | 0.733 | 0.733 | PASS |
| Section 5.8 | ordinal tree test auc | 0.73 | 0.73 | PASS |
| Section 5.8 | ordinal tree test cost | 121 | 121.0 | PASS |
| Section 5.8 | ordinal tree size | 3, 8 | 3, 8 | PASS |
| Section 5.8 | ordinal tree CV F1 | 0.565 | 0.565 | PASS |
| Section 5.8 | LR without imbalance treatment: specificity | 0.879 | 0.879 | PASS |
| Section 5.8 | LR without imbalance treatment: AUC | 0.805 | 0.805 | PASS |
| Table 5.11 | no checking account odds ratio | 0.37 | 0.37 | PASS |
| Table 5.11 | overdrawn account odds ratio | 1.68 | 1.68 | PASS |
| Table 5.11 | foreign worker odds ratio | 1.62 | 1.62 | PASS |
| Table 5.11 | male single odds ratio | 0.59 | 0.59 | PASS |
| Section 5.9.2 | Gini importance of no checking account | 0.411 | 0.411 | PASS |
| Section 5.9.2 | Gini importance of duration | 0.124 | 0.124 | PASS |
| Section 5.9.3 | LR permutation importance, checking account | 0.098 | 0.0977 | PASS |
| Section 5.9.3 | tree permutation importance, checking account | 0.162 | 0.1616 | PASS |
| Section 5.9.3 | LR permutation importance, purpose | 0.033 | 0.0326 | PASS |
| Section 5.9.3 | fairness-sensitive attributes each below 0.01 for LR | 1 | 1.0 | PASS |
