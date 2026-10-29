# Clinical Trial Keyword Recommendation

This project investigates machine learning methods for recommending keywords for clinical trial records using data from ClinicalTrials.gov.

## Models

* Frequency baseline
* Content-based filtering
* Collaborative filtering
* Hybrid recommendation
* Experimental field-weighted and frequency-adjusted recommenders

## Evaluation

Models are evaluated using Precision@5, Recall@5, and NDCG@5. The existing evaluation found content-based filtering performed best among the four original models.

## Dataset

The current collected dataset contains 1,000 clinical trials, of which 645 have recorded keywords.

## Setup

Install dependencies with: python -m pip install -r requirements.txt

## Responsible Use

This is an educational research prototype. Recommendations measure recovery of recorded keywords and do not establish clinical validity.

## New Algorithm: BM25



BM25 (Best Matching 25) is a ranking algorithm used to estimate the relevance of documents to a query.



In this project, BM25 ranks clinical trial descriptions based on their textual relevance. Keywords from the most relevant training trials are then aggregated to recommend keywords for a target trial.



### BM25 Evaluation Results



| Metric      | TF-IDF |   BM25 |

| ----------- | -----: | -----: |

| Precision@5 | 0.0180 | 0.0180 |

| Recall@5    | 0.0676 | 0.0676 |

| NDCG@5      | 0.0614 | 0.0675 |



In this experiment, BM25 achieved the same Precision@5 and Recall@5 as TF-IDF, with a higher NDCG@5. Further experiments are needed to determine whether this improvement generalizes.



### Run BM25



Generate recommendations:



```bash

python src/bm25_recommender.py

```



Evaluate BM25:



```bash

python src/evaluate_bm25.py

```



Compare evaluation results:



```bash

python src/compare_bm25_tfidf.py

```



Results are saved in the `results` directory.



**Limitation:** This is an educational research prototype. The evaluation measures recovery of recorded keywords, not clinical validity or medical correctness.





