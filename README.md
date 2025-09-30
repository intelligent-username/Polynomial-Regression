# Polynomial Regression From Scratch

![Cover](img.jpg)

---

## Outline

- [Polynomial Regression From Scratch](#polynomial-regression-from-scratch)
  - [Outline](#outline)
  - [Motivation \& Theory](#motivation--theory)
    - [The Optimization](#the-optimization)
      - [Setup (Single Feature)](#setup-single-feature)
      - [Remarks](#remarks)
      - [Worked numeric sketch (degree 2)](#worked-numeric-sketch-degree-2)
  - [Multivariate polynomial regression (generalization)](#multivariate-polynomial-regression-generalization)
  - [When to use polynomial regression](#when-to-use-polynomial-regression)
  - [License](#license)

---

## Motivation & Theory

*Note: for context, you might want to read up on [Linear Regression](https://github.com/intelligent-username/Linear-Regression) first. There's a lot of overlap/repetition in the concepts

Often times, we have a set of data points and want to generalize them to make useful predictions. This is largely what machine learning is about. Now, after engineering, cleaning, and transforming the data, you want to use it to train a model. However, the relationship between the features and the target variable may not always be linear.

Polynomial regression extends linear regression by adding extra predictors, obtained by raising each of the original predictors to a power. This allows the model to fit a wider range of curves and capture more complex relationships in the data.

Firstly, we have to define how we measure accuracy. The function that tells us how well our model is doing is called the **Loss Function**. Since this is our measure of accuracy, this is the metric we have to optimize for in order to create the ideal model.

In this writeup, we will be using the Mean Squared Error (MSE) as our loss function, which is defined as:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

If we're modelling a polynomial of degree $d$, our hypothesis function $h(x)$ can be expressed as:

$$
h(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d
$$

Here:

- $d$ is the degree of the polynomial.
- We have $(d + 1)$ coefficients to learn, written as

$$\beta_0, \beta_1, \ldots, \beta_d$$

- $x$ is the input feature.
- $h(x)$ is the predicted output.

Now, notice that this $d$-degree graph is only working in two dimensions (i.e. mapping one feature and one target variable). Expanding the scope to multiple features is straightforward, so I'll work with the single feature case for building the intuition and then proceed to generalize.

### The Optimization

This section derives the normal equations for polynomial regression from first principles for the single-feature (2‑D point) case, then generalizes to multiple features.

#### Setup (Single Feature)

Given data points $x_i, y_i$ for $i=1,\dots,n$ and a polynomial degree $d$, define the prediction, $\hat{y}_i$:

$$
\hat{y}_i = f(x_i) = \sum_{k=0}^d \beta_k x_i^k.
$$

With Loss:

$$
L(\beta) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
= \frac{1}{n} \sum_{i=1}^n \Big(y_i - \sum_{k=0}^d \beta_k x_i^k\Big)^2.
$$

This is a quadratic form in the parameter vector.
$$\beta=\begin{pmatrix}\beta_0\\\beta_1\\\vdots\\\beta_d\end{pmatrix}.$$

Now, in order to maximize accuracy, we want to minimize the loss. Since **MSE** is a degree two function that opens upwards, it's guaranteed to have a global minimum. To find it, compute the gradient and set to zero:

$$
\nabla L = \begin{pmatrix}
\frac{\partial L}{\partial \beta_0} \\
\frac{\partial L}{\partial \beta_1} \\
\vdots \\
\frac{\partial L}{\partial \beta_d}
\end{pmatrix} = \begin{pmatrix}
-\frac{2}{n} \sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) \\
-\frac{2}{n} \sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i \\
\vdots \\
-\frac{2}{n} \sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i^d
\end{pmatrix}
$$

Extrema occur when the derivative is zero, so set the gradient to zero:

$$
\sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i^j = 0
$$

\*Note:
$$
j \in \{0,\dots,d\}, j \in \mathbb{N}
$$

Rearrange the sums to isolate $\beta$:

$$
\sum_{i=1}^n y_i x_i^j = \sum_{k=0}^d \beta_k \sum_{i=1}^n x_i^{j+k}
$$

This is a linear system in the coefficients $\beta_k$. Define the $(d+1)\times(d+1)$ matrix $A$ and vector $b$:

$$
A_{j,k} = \sum_{i=1}^n x_i^{j+k},\qquad b_j = \sum_{i=1}^n y_i x_i^j,
$$
for $j,k=0,\dots,d$. The normal equations are

$$
A\beta = b.
$$

Now, notice how A can be expressed as a matrix product. Define the $n\times(d+1)$ design matrix $X$ with rows corresponding to data points and columns to powers of $x$:

$$
X = \begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{bmatrix}_{n\times(d+1)}.
$$

Then

$$
A = X^\top X,\qquad b = X^\top y,
$$
so the compact normal equation is

$$
X^\top X\beta = X^\top y,
$$
and the closed form solution (when $X^\top X$ is invertible) is

$$
\boxed{\beta = (X^\top X)^{-1} X^\top y}.
$$

This is identical to the linear regression formula. The only difference is what the design matrix $X$ looks like.

#### Remarks

- If the number of points is less than the degree plus one ($n < d+1$), the system has infinitely many solutions, so the model can't be pinned down.
- If the degree of the polynomial is very high, we risk overfitting the data and 'learning' the noise rather than the trend. **Complexity Control Techniques** like regularization, degree selection, and feature scaling, reduce overfitting by reducing complexity.
- While the above equation gives a closed form solution, in practice it's better to use numerical linear methods. For the sake of this learning endeavour, we'll be working with 'pure' polynomial regression.

#### Worked numeric sketch (degree 2)

Points: $(1,1), (2,4), (3,9)$. Fit $y=\beta_0+\beta_1 x+\beta_2 x^2$.

Design matrix

Notice that this is from the graph of $y=x^2$, with
$$x_1=1, x_2=2, x_3=3. $$

$$
X= \begin{bmatrix}1^0&1^1&1^2\\2^0&2^1&2^2\\3^0&3^1&3^2\end{bmatrix} = \begin{bmatrix}1&1&1\\1&2&4\\1&3&9\end{bmatrix},\quad y=\begin{bmatrix}1\\4\\9\end{bmatrix}.
$$

Compute $X^\top X$ and $X^\top y$:

$$
X^\top X=\begin{pmatrix}3 & 6 & 14 \\
6 & 14 & 36 \\
14 & 36 & 98\end{pmatrix},\qquad X^\top y=\begin{pmatrix}14\\36\\98\end{pmatrix}.
$$

Solve $X^\top X\beta=X^\top y$.

We have:

$$
X^\top X , \beta = X^\top y.
$$

Explicitly:

$$
\begin{bmatrix}
3&6&14 \\
6 & 14 & 36 \\
14 & 36 & 98
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\beta_2
\end{bmatrix}
\.=
\begin{bmatrix}
14 \\
36 \\
98
\end{bmatrix}
$$

By row reduction, we find:

$$
\beta_0 = 0,\ \beta_1 = 0,\ \beta_2 = 1.
$$

Thus:

$$
\beta =
\begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}.
$$

Which is the exact solution, as it forms $y = 0 + 0\cdot x + 1\cdot x^2 = x^2$.

---

## Multivariate polynomial regression (generalization)

Single‑feature polynomial regression generalizes by replacing the scalar power basis with a chosen set of multivariate basis functions (monomials or other basis) $\phi_j(\mathbf{x})$. For $p$ input features and maximum total degree $d$, the monomial basis contains all terms of the form

$$
\prod_{r=1}^p x_r^{\alpha_r},\qquad \text{with } \sum_{r=1}^p \alpha_r \le d.
$$

The number of such terms (including the constant) is

$$
N(p,d) = {p + d \choose d},
$$
which grows combinatorially and explains the rapid blow‑up in model complexity.

Construct the design matrix with columns $\phi_0(\mathbf{x}),\phi_1(\mathbf{x}),\dots,\phi_{N-1}(\mathbf{x})$ evaluated at each data point. The normal equations and solution are identical:

$$
\beta = (X^\top X)^{-1} X^\top y.
$$

Practical notes:

- **Interactions**: multivariate polynomials include interaction terms (e.g. $x_1 x_2$). Decide whether to include all cross terms or restrict to separable powers.
- **Feature explosion**: with moderate $p$ and $d$ the number of columns becomes large. Use feature selection, orthogonal bases, or sparse methods.
- **Regularization**: essential in multivariate cases to control variance. Ridge or Elastic Net are common.

---

## When to use polynomial regression

- When the relationship is smooth and globally well approximated by a low‑degree polynomial.
- When interpretability of polynomial coefficients is useful.

Avoid when:

- The relationship is locally complex (use splines or kernel methods).
- Dimensionality is high and interactions are many (use parsimonious models or regularized learners).

---

## License

This project is under the MIT License. Use and adapt freely with attribution.
