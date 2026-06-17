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
```

### 3. Accuracy part:
if you se the code the accuracy is cloise 85% and you might wonder why is not shooting up to 99% since the data is synthetically generated
the reason is the target column in there i use three & operatorss meaning i built a 3D rectangular box in my feature space every single inside the datapoint is 1 otherwise -1 since it is a linear classifier it is governed by the equation 
$$
w_1x_1 + w_2x_2 + w_3x_3 + b = 0
$$
it creates a perfect piece flat paper(hyperplane)
if the data requires the model to draw corners.It needs to say, "Stop going right at Age 30, turn 90 degrees, and go up until Salary
30,000."
the paper cannot bend 90° or curve at corners. the 15% error rate is the exact corners the paper cannot curve to hence it leaves them out
