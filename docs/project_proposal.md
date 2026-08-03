# ComplaintIQ Project Proposal
#Mines/Classes/dsci570/final-project

# ComplaintIQ
### A triage engine for consumer finance complaints

Name: Lorin Dawson
Section: 4
Dataset: [CFPB Consumer Complaint Database (US Consumer Financial Protection Bureau)](https://www.consumerfinance.gov/data-research/consumer-complaints/)

---
<div style="page-break-after: always;"></div>

# Proposal

### Problem statement
Financial institutions are drowning in complaints. The public CFPB database alone holds ~16.5M as of July 2026. A minority end in monetary relief to the consumer — those are the cases operations and compliance teams most need to catch early.

Late triage means slower remediation, unhappy customers, and regulatory risk. Today's process is manual and first-come-first-served.

ComplaintIQ is an ML product that scores each complaint for relief risk at intake, so reviewers prioritize the cases that matter most.

### Who will use it, and why
Primary buyer: operations or compliance teams at banks, credit bureaus, debt collectors, or fintechs. They pay because ComplaintIQ turns an undifferentiated inbox into a ranked queue. Analysts spend their hours on the highest-risk complaints first.

Secondary users: consumer advocacy groups and regulators, who use the theme clustering to spot emerging issues.

The product is decision support, not auto-decide. ComplaintIQ augments human reviewers.

### Current solutions
Most teams use manual routing, keyword rules, or product-based buckets. These are brittle, ignore the narrative where signal lives, and don't output a calibrated score.

ComplaintIQ improves by learning the narrative and metadata jointly, then outputting an interpretable risk score with the features driving it. Reviewers see *why* a complaint flagged, not just that it did.

# Proposed solution

Start simple and interpretable; add complexity only if it pays:
1. Logistic regression baseline: TF-IDF from narratives + one-hot intake metadata (product, issue, submitted via, state).
2. Tree-based / gradient-boosted models for non-linear interactions.
3. Unsupervised clustering over narratives for themes and nearest-neighbor retrieval.

**Evaluation:** Relief is rare (imbalanced target), so accuracy is useless. Evaluate on metrics built for rare positives and ranking:
- **PR-AUC** (primary), **ROC-AUC** (secondary).
- **Top-k lift** at the top 5–10% a team can actually review. Maps directly to value.
- **Calibration** (Brier score) so risk scores are trustworthy probabilities.
- **Time-based split** (train on older, test on recent months) to mirror real deployment.

The bar is modest: top-10% queue at 3–5× lift over base rate already saves hours and beats the current product-bucket heuristic.

### Data

The CFPB Consumer Complaint Database: ~16.5M complaints as of July 2026, refreshed continuously. A C++ ETL using Apache Arrow and Apache Parquet normalizes the raw export into an 18-column, analysis-ready Parquet.

Each row is one complaint:
- **Intake metadata:** product, sub_product, issue, sub_issue, company, state, zip_code, submitted_via, tags, date_received.
- **Narrative:** complaint_text (free text), has_narrative, complaint_text_length (derived).
- **Resolution:** company_response_to_consumer → monetary_relief label (1 = "Closed with monetary relief", else 0).

### Training & testing
Use only information available at intake. Exclude post-resolution fields (`company_response_to_consumer`, `timely_response`) to prevent leakage. 

Train/test split is chronological (older → recent months), so evaluation reflects real deployment without leakage.

Only 23% of complaints carry narratives. TF-IDF text features apply to that subset; the rest fall back to metadata-only scoring. Every complaint gets scored either way.

---
<div style="page-break-after: always;"></div>

# Additional information

### Timeline & milestones

The two graded final deliverables anchor the schedule: the video (Aug 5) and the
written report (Aug 7). Milestones below list what must be done ahead of each.

| Milestone                                                    | Target date  |
|--------------------------------------------------------------|--------------|
| Proposal submitted; ETL pipeline (raw → Parquet) validated   | Jul 18, 2026 |
| Exploratory data analysis complete (supervised + unsupervised) | Jul 21, 2026 |
| Baseline model: logistic regression + TF-IDF, time based split, PR-AUC | Jul 25, 2026 |
| Model iteration: tree / gradient boosted, imbalance handling, calibration | Jul 30, 2026 |
| Results locked: top k precision / lift evaluation + interpretability signals | Aug 2, 2026  |
| Video recorded and edited; report draft underway             | Aug 4, 2026  |
| Final Part 1 — Video submittal (hard deadline)               | Aug 5, 2026  |
| Report finalized; peer reviews of classmates' videos completed | Aug 7, 2026  |
| Final Part 2 — Written report (hard deadline)                | Aug 7, 2026  |

### References

1. Consumer Financial Protection Bureau, Consumer Complaint Database.
   https://www.consumerfinance.gov/data-research/consumer-complaints/
2. CFPB CCDB field reference. https://cfpb.github.io/api/ccdb/fields.html
3. scikit-learn: precision-recall / average precision (PR-AUC).
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
4. scikit-learn: TfidfVectorizer.
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. scikit-learn: resampling utility (`sklearn.utils.resample`) for over/undersampling. https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
6. scikit-learn: TimeSeriesSplit (time-based validation).
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
