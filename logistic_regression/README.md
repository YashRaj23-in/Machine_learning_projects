# Logistic Regression (L2 Regularization): First Principles


This directory contains a ground-up implementation of Logistic Regression with an L2 penalty (Ridge Regularization). The primary objective was to bypass high-level optimization solvers (like `scikit-learn` or `SciPy`) to manually code the log-loss objective function, gradient descent, and mathematical regularization using pure `NumPy`.

## The Mathematics of L2 Regularization
When translating the L2 penalty into code, most standard implementations hide the underlying calculus. This model explicitly handles two critical mathematical constraints to ensure stability:

### 1. The Calculus Trap (The Factor of 1/2)
A common error in from-scratch implementations is defining the penalty simply as the sum of squared weights ($\lambda \sum w^2$), which yields a derivative of $2\lambda w$. To ensure the gradient update remains mathematically clean and correctly scaled, the loss function explicitly multiplies the penalty by $0.5$:

$$J(\mathbf{w}) = \text{LogLoss}(\mathbf{w}) + \frac{\lambda}{2} \sum_{j=1}^{d} w_j^2$$

This perfectly cancels the $2$ during derivation, resulting in a clean $+\lambda w$ addition to the base gradient.

### 2. Isolating the Bias Term ($w_0$)
Penalizing the intercept term ($w_0$) mathematically forces the model's decision boundary toward the origin, destroying accuracy. This implementation utilizes strict `NumPy` array slicing to sever the bias term before calculating the penalty, and concatenates it back as a $0$ during the gradient update:

```python
# Penalty applied strictly to features w_1 through w_d
grad = base_grad + lambda_val * np.concatenate(([0], w[1:]))
