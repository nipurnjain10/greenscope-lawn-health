# Experiment analysis

This note is generated from `reports/model_metrics.json`. It is the same evaluation the research page reads.

The primary 10,000 rows were stratified by the simulated risk label into 7000 training, 1500 validation, and 1500 test rows (seed 42). Validation metrics come from models fit on training rows only. Test and robustness metrics come from the refit on training plus validation. The same row identifiers were used for every model.

All figures in this comparison are from the same untouched test partition (n = 1500), after the models were refit on training plus validation. The highest test accuracy is SVM (RBF) at 86.8% (macro-F1 86.9%). Soft voting accuracy is 86.0%. It does not beat SVM (RBF) (86.8%) on this test set. Equal-weight averaging was still reported because it was the pre-specified ensemble, not because the test set selected it. Soft voting misses 46 of 500 simulated high-risk cases, including 0 that it calls low risk. Accuracy alone hides that difference. These percentages describe the synthetic benchmark only.

Recall is uneven. The weakest class for soft voting is Moderate (79.2%) and the strongest is High (90.8%). High-risk precision is 89.7% and high-risk recall is 90.8% on 500 test rows. A missed high-risk case is the error a follow-up plan should not ignore. Class-wise scores use the simulated labels.

The robustness file was generated with a different seed, a hotter and drier latent climate, a warmer grass mix, more traffic, less rain, higher laboratory missingness, and the original score cutoffs. It does not reuse test rows, and its class mix differs from the balanced test set, so accuracy and macro-F1 need not move together. Logistic regression accuracy 80.6% to 83.7% (+3.0 points), macro-F1 80.7% to 79.3% (-1.3 points); Random forest accuracy 74.9% to 79.5% (+4.6 points), macro-F1 75.0% to 73.7% (-1.3 points); Gradient boosting accuracy 84.3% to 86.0% (+1.7 points), macro-F1 84.3% to 82.2% (-2.2 points); SVM (RBF) accuracy 86.8% to 87.8% (+1.0 points), macro-F1 86.9% to 84.5% (-2.4 points); Soft voting accuracy 86.0% to 87.8% (+1.8 points), macro-F1 86.0% to 84.5% (-1.5 points). On accuracy, soft voting is higher on the shifted sample. The comparison describes this synthetic shift. It is not an estimate of performance under real climate change.

Full soft voting has test macro-F1 86.0%. Leave-one-out averages use the same fitted models and do not retrain the others. without Logistic regression: macro-F1 86.7% (+0.7 points); without Random forest: macro-F1 86.8% (+0.7 points); without Gradient boosting: macro-F1 85.7% (-0.3 points); without SVM (RBF): macro-F1 83.0% (-3.0 points). Removing SVM (RBF) lowers macro-F1 by more than half a point, so those models contribute something the others do not fully replace on this test set.

For soft voting, the one-versus-rest macro expected calibration error is 0.090 and the multiclass Brier score is 0.239, both computed on the test predictions. A reliability point near the diagonal means that, inside that probability bin, the simulated label occurred about as often as the probability suggested. Empty bins are omitted. These quantities were not used to refit or recalibrate the models. They do not make the probabilities into real-world confidence.

Permutation importance is the drop in test accuracy when a column is shuffled. The largest mean drops for this model are soil moisture pct (23.3 accuracy points), drainage score (16.1 accuracy points), temperature c (11.3 accuracy points), phosphorus mg kg (7.7 accuracy points), soil compaction score (6.5 accuracy points). A large drop means the fitted model depends on that column in this synthetic test set. It does not mean the column caused the lawn condition, and it can give credit to a proxy for a latent variable the model never saw.

## Test metrics

| Model | Accuracy | Macro-F1 | Weighted-F1 | High called low |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 80.6% | 80.7% | 80.7% | 0 |
| random_forest | 74.9% | 75.0% | 75.0% | 0 |
| gradient_boosting | 84.3% | 84.3% | 84.3% | 0 |
| svm_rbf | 86.8% | 86.9% | 86.9% | 0 |
| soft_voting | 86.0% | 86.0% | 86.0% | 0 |
