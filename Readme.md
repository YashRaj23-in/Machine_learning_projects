# Machine Learning from First Principles

This repository contains ground-up implementations of foundational Machine Learning algorithms. 

**The strict development constraint:** No high-level machine learning APIs (like `scikit-learn` or `TensorFlow`) are used for model architecture, loss calculation, or optimization. All core algorithms are implemented translating mathematical objective functions and gradients into raw, vectorized code using pure `NumPy` and linear algebra.

## Implemented Models

| Algorithm | Conceptual Focus | Implementation |
| :--- | :--- | :--- |
| **Logistic Regression (L2)** | Calculus of Ridge Regularization, Gradient Averaging, Vectorized Loss | `[/logistic_regression]` |
| **Naive Bayes Classifier** | Conditional Probability, Matrix Addition over Iteration, Feature Correlation | `[/naive_bayes]` |
