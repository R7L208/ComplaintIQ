# ComplaintIQ, Findings

The "so what" from the notebooks in one place. Every number is measured (not an estimate). See the [glossary](GLOSSARY.md) for any unfamiliar terms or metrics.

**Setup:** predict `monetary_relief` (money back to the consumer). Base rate is **1.28%** overall
and **0.34%** in the recent-months test window; only **23%** of complaints carry a narrative. All
models use a leakage-safe, chronological split (train on older complaints, test on recent months).

---

## 1. Accuracy won't work; use ranking metrics
Relief is rare. "Always predict no relief" scores **~98.7% accuracy** while catching zero relief cases. Judge on **PR-AUC** and **top-k lift** instead. *(04a, 04b)*

## 2. The product-bucket heuristic beats the raw base rate
Rank complaints by their product's historical relief rate (no ML). This reaches **PR-AUC ~0.11**, and read at several queue depths: **9.7x lift at top-10%, 19.2x at top-5%, 29.4x at top-1%.** A learned model has to beat *this* baseline, not the raw base rate. Computing it (not just assuming it's weak) is what makes "beat the baseline" real. *(04a §product-bucket, 02 §9a)*

## 3. Top-10% lift hits a ceiling; PR-AUC keeps rising
| Model | PR-AUC | top-10% lift |
|---|---|---|
| Product-bucket heuristic | 0.11 | 9.7× |
| Logistic regression, raw metadata | 0.16 | 9.7× |
| Engineered metadata (target/freq encoding, interactions, text-shape, date) | 0.21 | 9.9× |
| Full-data Spark MLlib, engineered (16.5M rows) | 0.214 | 9.68× |

Engineering boosts **PR-AUC**. **Top-10% lift is mathematically capped at 10x** (lift = precision ÷ base rate ≤ 1 ÷ 0.1 = 10x). The model sits at ~97% of that ceiling. "Flat" lift isn't the model failing; it's the metric hitting its limit. *(04b, 06, 07a)*

## 4. Read a smaller queue to see what the model can do
Same model, different queue depths, with the heuristic baseline for contrast:

| Queue | Learned lift | Heuristic lift | Ceiling |
|---|---|---|---|
| top-10% | 9.7× | 9.7× | 10× |
| top-5% | 18.8× | 19.2× | 20× |
| **top-1%** | **59.7×** | 29.4× | 100× |

**"Ceiling" is the maximum lift arithmetically possible at that queue depth, not a model limit.** Lift = precision ÷ base rate, and precision can't exceed 1.0, so at the top-k fraction lift is capped at 1/k (top-10% → 10×, top-5% → 20×, top-1% → 100×). Reading the table against the ceiling is the point: at top-10% the models hit 9.7× against a 10× ceiling, so they're at ~97% of the most that's *possible* there. There is no room left to look better, regardless of how good the model is. The heuristic and learned model therefore tie at 10% and 5% (both near-saturated). Only at **top-1%**, where the ceiling is a distant 100×, is there headroom to separate them, and the learned model does: **59.7× vs 29.4×**. That is exactly the shallow queue reviewers actually work, where ~**1 in 5** flagged complaints is a relief case (vs. 0.34% random). The operating point (how deep reviewers work) matters more than tuning the model. Every notebook now reports lift at top-1%, 5%, and 10%. *(07a, 02, 04)*

## 5. Sampled results hold at full scale
Engineered model on 300k stratified sample: PR-AUC 0.21. Same model on full 16.5M rows with Spark MLlib: **0.214**. Nearly identical. Validates the sklearn-on-sample workflow for fast iteration. *(06 vs 07a)*

## 6. Representation choice is task-dependent (the headline result)
GPU serverless (NVIDIA A10G):

| Task | TF-IDF (word matching) | Sentence embeddings (meaning) |
|---|---|---|
| **Unsupervised, theme clustering** (ARI vs product × issue) | 0.014 | **0.067, embeddings win ~5×** |
| **Supervised, relief prediction** (PR-AUC, narrative subset) | **0.271, TF-IDF wins** | 0.241 |
| Supervised, TF-IDF **+** embeddings stacked |, | **0.281 (best); top-1% lift 6.1×** |

The best representation depends on the task. Theme clustering rewards meaning (embeddings win). Relief prediction is driven by word choice (TF-IDF wins). Embeddings help only as a complement (stacked), not a replacement. *(08, 09, 10)*

## 7. Larger embedding samples don't improve clustering; full-corpus is impractical here
Embedding clustering was re-run on GPU serverless at 250k (ARI **0.056**, NMI 0.232), and this stayed **below the 8k-sample's 0.067**: a bigger sample did not buy better themes. Scaling to the full 3.8M in-memory KMeans is impractical on Free Edition serverless (the job runs for hours; the fitted MLlib model also exceeds Spark Connect's 256MB transfer cap, see the 07b note). So the practical result is: embeddings beat TF-IDF for clustering, but more rows past a small sample add cost without quality. *(10)*

## 8. The unsupervised side is the weaker half, and where the headroom is
Best clustering (ARI ~0.067) only weakly recovers the known theme structure. MinHash/LSH, LSA, and embeddings each help incrementally, but this is the honest "not solved" part of the project. Clear place for future work. *(05, 06, 08)*

---

## Caveats (read the numbers with these in mind)
- **Finding 6 embedding comparison is on a small narrative subset** (~5,600 test rows). Treat lift figures as directional, not precise.
- **Read cluster purity alongside ARI/NMI, never alone.** Purity rises monotonically with the number of clusters, so it does not correct for chance the way ARI/NMI do. (An earlier k×k assignment in notebook 10 silently dropped most true labels and produced a wrong number; it now uses the standard per-cluster majority over the full contingency table. ARI/NMI remain the headline clustering metrics.)
- Finding 6's clustering ARI is from an 8k sample (08); finding 7's is the full corpus (10). Same direction (embeddings > TF-IDF), different experiments.
- All lift/PR-AUC numbers use the chronological recent-months test window. Different splits shift the absolute values.
- **Reproduced end-to-end on Databricks Free Edition serverless (2026-08-03).** The full pipeline (01-10) ran as a bundle job; 07a reproduced PR-AUC 0.214 and top-1% lift 59.7x on the full 16.5M rows, and 08/09/10 reproduced the representation findings (09: TF-IDF 0.271, embeddings 0.241, stacked 0.280). One exception: **07b (full-corpus MLlib KMeans) does not run on serverless** - the fitted model exceeds Spark Connect's 256MB transfer cap regardless of sample size or feature dimension (locally the saved model is ~1MB, so it is a Connect serialization artifact, not the model itself). Run 07b on classic Spark; the clustering story is carried by 05/08/10.
