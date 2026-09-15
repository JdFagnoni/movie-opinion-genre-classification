# Movie Opinion Classification

Binary classification of audience opinion (positive/negative) on movies from structured metadata, comparing 5 ML models under three independent validation schemes to pick a model for a class Kaggle competition.

## Overview

This is the opinion-classification half of a two-part Machine Learning assignment for *Machine Learning para Sistemas Inteligentes* at Universidad ORT Uruguay (July 2025). Given structured metadata about ~4,800 movies (budget, revenue, runtime, popularity, vote count, genre, language, director, release status), the notebook predicts a binary `opinion` label that is close to perfectly balanced (50.6% / 49.4%). No natural-language fields (`overview`, `keywords`, `original_title`) are used for this task — they're dropped after cleaning, since this notebook works purely with structured/tabular features.

What makes it worth a second look technically is the evaluation harness rather than any single model: every one of the five algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP) is run through the *same* three validation strategies — a single stratified holdout, a 10x repeated holdout, and 5-fold stratified cross-validation with `GridSearchCV` — using shared helper functions. That makes it possible to see, per model, how much a single-split score can overstate performance (e.g. Logistic Regression's holdout F1 of 0.716 drops to 0.698 under repeated holdout) before trusting a hyperparameter search built on top of it.

The final model (Gradient Boosting) was submitted to a class-run Kaggle competition, along with Random Forest and a tuned Logistic Regression as backups, plus several weighted ensembles of Gradient Boosting + Random Forest — none of which beat the single Gradient Boosting model.

**Note on scope:** the original assignment also included a second, separate notebook (`prediccionGenero`) that predicts `main_genre_top10` using NLP techniques (text cleaning, stopword removal, TF-IDF vectorization). That work is described in `Documentation.pdf` (sections 2.1–2.10) but the notebook itself is not included in this repository — only the opinion-classification notebook below is. Numbers for that part are not reproduced here since they can't be verified against runnable code in this repo.

## Architecture / Approach

Pipeline implemented in [opinion_prediction.ipynb.ipynb](opinion_prediction.ipynb.ipynb):

1. **Load** `movies_train.csv` (3,602 rows) and `movies_test.csv` (1,201 rows), 14 columns each, from Google Drive (Colab).
2. **Clean**: parse `release_date` and impute missing dates with the median; impute missing numeric fields (`budget`, `revenue`, `runtime`, `popularity`, `vote_count`) with the column median; fill free-text fields with empty strings; fill low-cardinality categoricals with the mode; fill the high-cardinality `director` field with `"Unknown"`; drop duplicate rows.
3. **Feature engineer**:
   - Drop `id` and `title` (no predictive signal).
   - Group directors with fewer than 10 films into an `"Other"` bucket (1,943 unique directors → 22 categories), keeping the 21 most prolific directors (Spielberg, Scorsese, Eastwood, etc.) as their own category.
   - One-hot encode `original_language`, `status`, `main_genre_top10`, and the grouped director column (`drop_first=True`), producing 74 features.
4. **Split** 80/20 train/validation, stratified on `opinion`, `random_state=42`.
5. **Model & evaluate** (`df_clean`, numeric-only matrix): for each of Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and MLP —
   - Baseline (default/untuned hyperparameters) via holdout and 10x repeated holdout.
   - Hyperparameter search via `GridSearchCV` over `StratifiedKFold(5)`, scoring on F1 (refit) and AUC-ROC.
6. **Compare** all 5 tuned models on cross-validated F1 and AUC-ROC, select the best.
7. **Final fit**: retrain the selected model (Gradient Boosting) on the full training set, re-validate on the same 80/20 split for a final metrics snapshot.
8. **Kaggle submission**: apply identical preprocessing to `movies_test.csv`, align its one-hot columns to the training schema (filling missing dummy columns with 0, dropping extras), and generate probability predictions from Gradient Boosting, Random Forest, tuned Logistic Regression, and several GB/RF weighted ensembles.

## Key results

5-fold stratified cross-validation, after hyperparameter tuning (from the notebook's model-comparison output):

| Model | F1 (CV) | AUC-ROC (CV) | Best hyperparameters |
|---|---|---|---|
| Gradient Boosting | **0.7294** | **0.8230** | `n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_split=5` |
| Random Forest | 0.7259 | 0.8127 | `n_estimators=200, max_depth=None, min_samples_split=5, criterion=gini` |
| Logistic Regression | 0.7076 | 0.7955 | `C=10, penalty=l2` |
| MLP | 0.7034 | 0.7983 | `hidden_layer_sizes=(100,), alpha=0.0001, learning_rate_init=0.001` |
| Decision Tree | 0.6866 | 0.7592 | `max_depth=5, min_samples_split=20, criterion=gini` |

**Selected model: Gradient Boosting.** Final retrain + held-out validation (80/20 split, same seed):

- Accuracy: **0.7614**, Precision: 0.7644, Recall: 0.7472, **F1: 0.7557**, **AUC-ROC: 0.8334**
- Confusion matrix: `[[283, 82], [90, 266]]` (283 true negatives, 266 true positives, 82 false positives, 90 false negatives)

**Kaggle competition** (class-run "MLSI Primer Semestre 2025", AUC-ROC scored): the Gradient Boosting submission reached a best public leaderboard score of **0.788 AUC-ROC**, ranked 14th (team *Fagnoni-Baraibar-DeFeo*) — per the leaderboard screenshot in `Documentation.pdf`. The GB+RF weighted ensembles tried afterward did not beat the single Gradient Boosting model.

## Design decisions

**High-cardinality `director` gets a frequency-threshold "Other" bucket, not mode imputation or one-hot-all.** Of 1,943 unique directors, 1,275 direct exactly one film in the training set. Two simpler alternatives were rejected: (a) imputing missing directors with the mode, which would silently bias the most common director's frequency upward and misrepresent one-off filmmakers as if they were a copy of whoever's most frequent; and (b) one-hot-encoding all 1,943 directors, which would blow up dimensionality relative to 3,602 training rows and mostly encode noise from directors with a single film. Instead, directors with fewer than 10 films are grouped into `"Other"`, cutting the category count to 22 while preserving a real, distinguishable signal for the directors who actually recur often enough to matter (Spielberg: 24 films, Eastwood: 17, etc.).

**Missing values are imputed differently depending on cardinality and skew, instead of one blanket strategy.** Skewed numeric fields (`budget`, `revenue`) are filled with the median rather than the mean, specifically because a handful of blockbuster budgets/revenues would drag the mean far from the typical value. Low-cardinality categoricals (`status`, `original_language`, `main_genre_top10`) are filled with the mode, since a small category set means the dominant category is genuinely likely to be correct. That same mode-fill approach is deliberately *not* applied to `director` (see above) — the cost of a wrong categorical fill scales with how many plausible categories there are, so the strategy had to change with cardinality rather than being applied uniformly across all categorical columns.

## Tech stack

- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling & evaluation**: scikit-learn — `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `MLPClassifier`, `StandardScaler`, `make_pipeline`, `train_test_split`, `StratifiedKFold`, `GridSearchCV`, and the `accuracy_score` / `precision_score` / `recall_score` / `f1_score` / `roc_auc_score` / `confusion_matrix` / `classification_report` metrics
- **Environment**: Google Colab + Google Drive (original execution environment; notebook mounts Drive to read the CSVs)

## How to run

The notebook was authored for Google Colab with the datasets stored on Google Drive. The raw datasets (`movies_train.csv`, `movies_test.csv`) are **not included in this repository** (see `.gitignore`) — you'll need your own copy to reproduce the results.

1. Clone the repo and create an environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Place `movies_train.csv` and `movies_test.csv` in the repo root (or wherever you point the paths).
3. In [opinion_prediction.ipynb.ipynb](opinion_prediction.ipynb.ipynb), replace the Google Drive mount cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```
   and the two `pd.read_csv("/content/drive/My Drive/...")` calls with local paths, e.g. `pd.read_csv("movies_train.csv")`.
4. Launch Jupyter and run all cells top to bottom (later cells reuse variables — like `rare_directors_list` and `X_test_final` — defined in earlier ones, so it isn't safe to run cells out of order):
   ```bash
   jupyter notebook opinion_prediction.ipynb.ipynb
   ```

## Team

Team project (3 members): Francisco Baraibar, Leandro De Feo, and Juan Diego Fagnoni. Submitted as the course deliverable for *Machine Learning para Sistemas Inteligentes*, Universidad ORT Uruguay, July 2025. This repository reflects the team's joint submission (notebook + `Documentation.pdf`); the original work was done collaboratively rather than split into separately attributable files.
