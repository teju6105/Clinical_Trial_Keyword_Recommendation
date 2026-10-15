# Clinical Trial Keyword Recommendation

This project investigates machine learning methods for recommending keywords for clinical trial records using data from ClinicalTrials.gov.

## Models
- Frequency baseline
- Content-based filtering
- Collaborative filtering
- Hybrid recommendation
- Experimental field-weighted and frequency-adjusted recommenders

## Evaluation
Models are evaluated using Precision@5, Recall@5, and NDCG@5. The existing evaluation found content-based filtering performed best among the four original models.

## Dataset
The current collected dataset contains 1,000 clinical trials, of which 645 have recorded keywords.

## Setup
Install dependencies with: python -m pip install -r requirements.txt`n
## Responsible Use
This is an educational research prototype. Recommendations measure recovery of recorded keywords and do not establish clinical validity.
