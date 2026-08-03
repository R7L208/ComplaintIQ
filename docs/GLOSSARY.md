# ComplaintIQ Glossary

A plain-language reference for the terms, methods, and metrics used here. Written so a non-specialist can follow the notebooks and results. Where a metric has a "good" and "bad" direction, it's stated explicitly.

Running example: ComplaintIQ predicts whether a complaint will end in **monetary relief** (money back to the consumer). Relief is **rare**, about **1.28%** of complaints (roughly 1 in 78), which is why imbalance dominates the notes below.

---

## 1. The problem setup

- **Complaint / narrative**, a consumer complaint record. The **narrative** is the free-text
 description the consumer wrote; only about **23%** of complaints include one.
- **Target (label)**, the thing we predict: `monetary_relief`, which is `1` if the complaint
 closed "with monetary relief" and `0` otherwise.
- **Base rate**, how common the positive outcome is overall. Here ~1.28%. It's the "do nothing clever" reference point everything else is compared against.
- **Class imbalance**, when one outcome vastly outnumbers the other (here, ~99% "no relief").
 Imbalance is why plain accuracy is misleading and why we lean on the metrics in section 5.
- **Supervised learning**, learning from labeled examples (we know the true `monetary_relief`
 for past complaints) to predict the label on new ones.
- **Unsupervised learning**, finding structure with **no** labels; here, grouping complaints
 into **themes** by their text (see clustering, section 6).
- **Feature**, an input the model uses to make a prediction (e.g. the product, the state, words
 in the narrative). **Feature engineering** is the craft of turning raw fields into useful
 features.
- **Leakage**, accidentally letting the model see information it would not have at prediction
 time. Example: `company_response_to_consumer` reveals the outcome, so using it as an input
 would be cheating. ComplaintIQ excludes such **post-resolution** fields on purpose.
- **Chronological split**, training on older complaints and testing on the most recent ones,
 instead of a random split. This mimics real deployment (predicting the future) and avoids
 leakage from "seeing" later data during training.
- **Baseline**, a deliberately simple model that a real model must beat to matter. The main one here is the **product-bucket heuristic**: rank complaints by their product's historical relief rate. If a fancy model doesn't beat that, it's not earning its keep.

---

## 2. Data & text methods

- **TF-IDF (Term Frequency–Inverse Document Frequency)**, turns text into numbers by weighting each word by its rarity and distinctiveness. Rare words get higher weight. Knows *which words co-occur*, not what they mean.
- **n-gram**, a sequence of *n* adjacent tokens. A **unigram** is one word ("fee"); a **bigram**
 is two ("late fee"). **Character n-grams** are letter sequences, robust to typos and redactions.
- **Sentence embedding**, a modern alternative: a pretrained model converts each narrative into a dense vector (~384 numbers) capturing *meaning*. "Charged a fee" and "billed me" land near each other despite no shared words. Better for theme grouping.
- **LSA / TruncatedSVD (dimensionality reduction)**, compresses a huge sparse TF-IDF matrix into a few hundred dense columns, keeping most signal. Makes distance-based methods work. Key: reduction *compresses* an existing representation; an embedding *learns* one, different steps.
- **Target (mean) encoding**, replaces a category (e.g. `company`) with its historical relief rate, one number instead of thousands of one-hots. Computed on training data only to avoid leakage.
- **Frequency encoding**, replaces a category with how often it appears; volume is itself a
 signal and it stabilizes rare categories.
- **One-hot encoding**, the naive alternative: one 0/1 column per category value. Simple but
 explodes for high-cardinality fields like `company` or `zip_code` (thousands of columns).
- **Stratified sample**, a sample that preserves the class proportions (so rare relief cases
 aren't lost). Used when a method can't process all 16.5M rows.
- **MinHash / LSH (Locality-Sensitive Hashing)**, fast way to find **near-duplicate** texts (e.g., templated complaints) without comparing every pair. Used to deduplicate before clustering so artificial copies don't form fake themes.
- **Deduplication**, removing exact or near-duplicate records so they don't distort results.

---

## 3. Models & algorithms

- **Logistic regression**, a simple, interpretable model that outputs a probability (0–1) for a
 yes/no outcome. ComplaintIQ's main supervised model.
- **KMeans**, a clustering algorithm that groups points into *k* clusters by nearest center. Used
 for theme discovery. *k* here is set to the number of products.
- **Class weighting (`class_weight="balanced"`)**, tells the model to pay proportionally more
 attention to the rare class, countering imbalance. A model setting, not a feature.
- **Calibration (Platt / isotonic)**, adjusts raw scores so a "0.30" means a 30% chance. Needed to combine two models (text model on the 23% with narratives, metadata model on the rest) into one fair ranking.
- **Spark / MLlib**, distributed engine (`pyspark.ml`) for training on the **full** 16.5M rows across a cluster, not a sample on one machine.
- **GPU serverless**, on-demand GPU (NVIDIA A10G) making embedding millions of narratives practical.

---

## 4. Evaluation concepts

- **Train / test split**, fit the model on one portion of data, measure it on a held-out portion
 it never saw. The test score is the honest one.
- **Overfitting**, when a model memorizes noise in the training data and does worse on new data.
 Training error always falls as a model gets more complex, but true (new-data) error follows a
 U-shape, the bottom of that U is the sweet spot.
- **Underfitting**, the opposite: a model too simple to capture the real pattern; high error
 everywhere.
- **Confusion matrix**, a 2×2 table of counts: true positives (TP), false positives (FP), false
 negatives (FN), true negatives (TN). Most classification metrics are derived from it.

---

## 5. Supervised metrics (with good/bad direction)

> For this project the metric that matters most operationally is **lift** at a small queue size,
> because reviewers work a ranked queue. PR-AUC is the primary ranking-quality metric; accuracy is
> explicitly *not* trusted here.

- **Accuracy** = (TP+TN) / all. **Higher looks better, but misleading here.** Predicting "no
 relief" for everyone scores ~98.7% accuracy while catching zero relief cases. Do not use on
 imbalanced data.
- **Precision** = TP / (TP+FP). Of the complaints flagged high-risk, the fraction that truly got
 relief. **Higher is better** (0–1). High precision = few false alarms.
- **Recall (sensitivity)** = TP / (TP+FN). Of all real relief cases, the fraction caught.
 **Higher is better** (0–1). High recall = few misses.
- **F1 score** = harmonic mean of precision and recall. **Higher is better** (0–1). Balances the
 two; useful under imbalance.
- **ROC-AUC**, ranking quality across all thresholds. **Higher is better**: **0.5 = random,
 1.0 = perfect.** Can look optimistically high when positives are rare, so pair it with PR-AUC.
- **PR-AUC (average precision)**, ranking quality focused on the rare positive class.
 **Higher is better** (0–1). A no-skill model scores about the **base rate** (~0.0128 here), so
 anything meaningfully above that is real signal. **Preferred over ROC-AUC when positives are
 rare.** (Project baselines: heuristic ~0.11, engineered models ~0.21–0.30.)
- **Top-k precision / lift**, the review-queue metric. Take the top *k*% of complaints the model
 ranks highest; **lift** = (relief rate in that slice) ÷ (base rate). **Higher is better; 1× means
 no skill.** *Important ceiling:* lift at top-*k*% can never exceed **1/k** (precision ≤ 1), so
 **top-10% lift maxes at 10×, top-5% at 20×, top-1% at 100×.** A smaller queue reveals ranking
 power that top-10% cannot express. (Project full-data model: ~9.7× at top-10%, ~19× at top-5%,
 ~60× at top-1%.)
- **Brier score**, squared error of the predicted probabilities. **Lower is better** (0 = perfect,
 worst case 1). Measures whether the probabilities themselves are trustworthy, not just the ranking.
- **Log loss (cross-entropy)**, like Brier, penalizes confident wrong probabilities. **Lower is
 better.**
- **Balanced accuracy**, average of per-class recall. **Higher is better; 0.5 = chance** regardless
 of class ratio. An imbalance-robust stand-in for accuracy.
- **MCC (Matthews correlation coefficient)**, a single balanced score using all four confusion
 cells. Range **−1 to +1; 0 = random, +1 = perfect.** Stays honest under heavy imbalance.

---

## 6. Unsupervised (clustering) metrics

> Clustering has **no labels**, so there is no "accuracy." We judge clusters by how well they line
> up with a **label-free yardstick**, the `product × issue` combination, which labels real
> complaint themes the model never saw. Two families: *external* (compare to that yardstick) and
> *internal* (judge shape alone).

- **`product × issue` yardstick**, the cross-tabulation of a complaint's product and issue. A few
 combinations dominate as coherent themes; a good clustering should roughly rediscover them. Used
 only to *measure* clustering, never as a model input (that would be circular).
- **ARI (Adjusted Rand Index)**, external: agreement between the clusters and the yardstick,
 corrected for chance. **Higher is better: 0 = random, 1 = identical** (can dip slightly negative).
 The main clustering score here. (Project: raw TF-IDF ~0.014; embeddings ~0.067, several× better,
 but still low in absolute terms, i.e. themes are only weakly recovered.)
- **NMI (Normalized Mutual Information)**, external: how much knowing the cluster tells you about
 the true theme. **Higher is better** (0–1).
- **Purity**, external: the fraction of points that sit in their cluster's most-common true theme.
 **Higher is better** (0–1). Simple and driver-free, but **not** chance-corrected, so it reads
 higher than ARI for the same clustering, don't compare a purity number to an ARI number.
- **Silhouette score**, *internal*: how tight and well-separated the clusters are, using the data
 alone. Range **−1 to +1; higher is better, ~0 = overlapping.** Caveat: it rewards clusters shaped
 the way the metric expects, so a high silhouette doesn't guarantee the clusters are *useful*, 
 treat it as one signal, not a verdict.

---

## 7. Quick "is this number good?" cheat sheet

| Metric | Range | Good direction | No-skill value |
|---|---|---|---|
| Accuracy | 0–1 | higher (but misleading if imbalanced) | = majority-class share (~0.987 here) |
| Precision / Recall / F1 | 0–1 | higher |, |
| ROC-AUC | 0–1 | higher | 0.5 |
| PR-AUC | 0–1 | higher | ≈ base rate (~0.013 here) |
| Top-k lift | ≥ 0 | higher (cap = 1/k) | 1× |
| Brier / Log loss | ≥ 0 | **lower** |, |
| Balanced accuracy | 0–1 | higher | 0.5 |
| MCC | −1 to +1 | higher | 0 |
| ARI | ≤ 1 | higher | 0 |
| NMI | 0–1 | higher | 0 |
| Purity | 0–1 | higher (not chance-corrected) | ≈ largest theme share |
| Silhouette | −1 to +1 | higher | ~0 |

---

*Metric definitions follow the DSCI570 course notes (Evaluation Metrics; Supervised/Unsupervised
Learning Evaluation). Project numbers are the measured results reported in the notebooks.*
