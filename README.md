# Naive Bayes Classifier: First Principles

## Overview
This repository contains a ground-up implementation of a Naive Bayes classification model. The strict development constraint for this project was to completely bypass high-level machine learning APIs like `scikit-learn`. 

The primary objective was to translate the underlying probability mathematics and decision rules into raw Python, focusing on computational efficiency and matrix operations.

Dataset Link:https://www.ssa.gov/oact/babynames/limits.html 
## The Technical Pivot: Vectorization over Iteration
The initial implementation utilized standard Pandas `.iterrows()` loops to calculate the conditional probabilities row-by-row. While functionally correct, it severely bottlenecked the computation. 

To align with high-performance computing standards, the core logic was refactored. The final inference engine strips away Python-level loops, mapping the probability dictionaries directly into pure `NumPy` arrays and executing the Likelihood calculations via vectorized matrix addition along the feature axis.

There was one more thing first i thought to increase the accuracy i should add more features and it actually increased but upon more in depth analysis i found that naive bayes divides the weighted probability among the features
equally and hence when i added a feature last_letter i already had a feature called last_twoletters there was no point to do so but did it the accuracy went up to 77.89% 
lets say there was previously 3 features then naive bayes weighted probability of those features was 1/3 but when i added the fourth feature the last_letter it essentially made the weighted probability of the 
last letter to 2/4 thats why the accuracy went up. the numbers 1/3 and 2/4 are for better understanding not how it really works under the hood and this what we call feature correlation.
## Feature Engineering
The model predicts gender based on engineered name features. The following parameters were extracted and vectorized for the probability matrices:

| Feature | Description | Data Type |
| `first_name` | The initial character of the name | String |
| `last_two_letters` | The final two characters of the name | String |
| `length` | Total character count of the name | Integer |
| `Vowels_Count` | Total count of [a, e, i, o, u] | Integer |

## Execution Requirements
* Pure NumPy for all core matrix operations and probability calculations.
* Pandas strictly reserved for initial data ingestion and feature engineering.

## How to Run
1. Clone the repository.
2. Ensure you have the required dataset (names dataset by year).
3. Update the folder path in the main execution block.
4. Run the script to train the model, evaluate accuracy, and test custom name inputs.
