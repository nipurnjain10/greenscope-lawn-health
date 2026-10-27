# Project overview

## Titles

- Course title: Using Machine Learning to Improve the Business of a Chain of Gardening Works Like The Grounds Guys.
- Technical title: An Explainable Ensemble Learning Framework for Plant Health Risk Prediction and Cost-Aware Landscaping Decisions.
- Application: GreenScope — Lawn Health & Service Planning.

GreenScope is an independent academic prototype for a five-member M.Tech Computer Science and Engineering group at IIT Roorkee. It is not developed, owned, approved, or affiliated with The Grounds Guys. The company name is the business context in the course title.

## Question

Can an ensemble of diverse supervised classifiers produce useful and explainable lawn-health risk assessments on a source-informed synthetic benchmark, and how can those assessments support transparent, cost-aware landscaping decisions?

The further questions are whether soft voting beats the best individual model on the held-out synthetic test set, which features the models depend on, whether a shifted environmental file changes performance, and how a risk class becomes a maintenance priority and an illustrative budget.

## What was built

- A generator for 10,000 synthetic inspections and a 2,000-row robustness file. One target, `overall_lawn_health_risk`, with classes Low, Moderate, and High. The same rows are used by every model.
- Four scikit-learn pipelines: logistic regression, random forest, gradient boosting, and a calibrated RBF support-vector machine.
- Equal-weight soft voting is the only ensemble. It is reported even when it is not the most accurate model.
- Permutation importance and a local leave-one-value-to-baseline explanation. SHAP is not used.
- Written water, nutrient, and drainage screens, actions, a follow-up interval, and editable illustrative prices.
- A Streamlit application with an operational assessment and a separate research view, including a demonstration that scores a real held-out test row.
- A report and a presentation generated from the saved metrics.

## What is not claimed

The labels are simulated. The test scores are not field accuracy. The prices are not a quotation and not evidence of savings. No customer record was used.

Names, enrollment numbers, and the group number are placeholders in `group_info.json` until the group fills them in.
