# Polynomial Regression

![Cover](img.jpg)

---

## Outline

- [Polynomial Regression](#polynomial-regression)
  - [Outline](#outline)
  - [Motivation \& Theory](#motivation--theory)
    - [The Optimization](#the-optimization)
      - [Remarks](#remarks)
      - [Example (degree 2)](#example-degree-2)
  - [Multivariate Polynomial Regression](#multivariate-polynomial-regression)
  - [When to Use](#when-to-use)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [License](#license)

---

## Motivation & Theory

*Note: for context, you might want to read up on [Linear Regression](https://github.com/intelligent-username/Linear-Regression) first. There's a lot of overlap.*

Say we want to deterministically create a predictive model with nice, clean data. However, even the most accurate straight line isn't accurate enough for most cases. Ecce Polynomial Regression.

Polynomial regression extends linear regression by expanding input(s) into polynomial basis functions and fitting a linear model in that basis. It is linear in the parameters, so ordinary least squares applies.

Firstly, we have to define [Loss Function](github.com/intelligent-username/Loss-Functions). Since this is our measure of accuracy, this is the metric we have to optimize.

In this writeup, we will be using the Mean Squared Error (MSE):

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
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
- $h(x)$ is the predicted output. Often written as $\hat{y}$.

Now, notice that this $d$-degree graph is only working in two dimensions (i.e. mapping one feature to a target). Expanding the scope to multiple features is straightforward, so I'll work with the single feature case for  building the intuition and then proceed to generalize.

### The Optimization

Given data points $x_i, y_i$ for $i=1,\dots,n$ and a polynomial degree $d$, define the prediction, $\hat{y}_i$:

$$
\hat{y}_i = f(x_i) = \sum_{k=0}^d \beta_k x_i^k.
$$

With Loss:

$$
L(\beta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \\
= \frac{1}{N} \sum_{i=1}^N \Big(y_i - \sum_{k=0}^d \beta_k x_i^k\Big)^2.
$$

With:

- $N$ = number of data points
- $d$ = degree of polynomial
- $x_i$ = input feature of point $i$
- $\hat{y}_i$ = predicted output for point $i$
- $y_i$ = label ('target') of point $i$
- $\beta_k$ = coefficient of $x^k$ in the polynomial

This is a quadratic function in $\beta$.

$$\beta=\begin{pmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_d \\
\end{pmatrix}.$$

Now, in order to maximize accuracy, we want to minimize the loss. Since **MSE** is convex, it's guaranteed to have a global minimum. To find it, compute the gradient:

$$
\nabla _\beta L = \begin{pmatrix}
\frac{\partial L}{\partial \beta_0} \\
\frac{\partial L}{\partial \beta_1} \\
\vdots \\
\frac{\partial L}{\partial \beta_d}
\end{pmatrix} = -\frac{2}{n}\begin{pmatrix}
\sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) \\
\sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i \\
\vdots \\
\sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i^d
\end{pmatrix}
$$

and set to zero:

$$
\sum_{i=1}^n \big(y_i - \sum_{k=0}^d \beta_k x_i^k\big) x_i^j = 0
$$
With $j \in \{0,\dots,d\} \land j \in \mathbb{N}$

Rearrange :

$$
\sum_{i=1}^n y_i x_i^j = \sum_{k=0}^d \beta_k \sum_{i=1}^n x_i^{j+k}
$$

This is a linear system in the coefficients $\beta_k$. To save space and generalize, define the $(d+1)\times(d+1)$ matrix $A$ and vector $b$:

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
\vdots & \vdots & \vdots & \ddots & \vdots \\
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

This is identical to the linear regression formula, except [Vandermonde](https://en.wikipedia.org/wiki/Vandermonde_matrix)-like design replaces the plain feature matrix.

#### Remarks

- Because the matrix is Vandermonde-like, its determinant can be computed efficiently, but numerical stability can be an issue for high degrees. However, this is a topic for another time.
- If the number of points is less than the degree plus one ($N < d+1$), the system has infinitely many solutions, so the model can't be pinned down.
- If the degree of the polynomial is very high, we risk overfitting the data and 'learning' the noise rather than the trend. **Complexity Control Techniques** like regularization, degree selection, and feature scaling, reduce overfitting by reducing complexity.
- While the above equation gives a closed form solution, in practice it's better to use numerical linear methods. For the sake of this learning endeavour, we'll be working with 'pure' polynomial regression.

#### Example (degree 2)

Points: $(1,1), (2,4), (3,9)$. Fit $y=\beta_0+\beta_1 x+\beta_2 x^2$.

Notice that this is from the graph of $y=x^2$, with
$$x_1=1, x_2=2, x_3=3. $$

Design matrix

$$
X= \begin{bmatrix}
1^0&1^1&1^2 \\
2^0&2^1&2^2 \\
3^0&3^1&3^2
\end{bmatrix}
= \begin{bmatrix}
1&1&1 \\
1&2&4 \\
1&3&9
\end{bmatrix},
\quad
y=\begin{bmatrix}
1 \\
4 \\
9
\end{bmatrix}.
$$

Compute $X^\top X$ and $X^\top y$:

$$
X^\top X=\begin{pmatrix}3 & 6 & 14 \\
6 & 14 & 36 \\
14 & 36 & 98\end{pmatrix},\qquad X^\top
y=\begin{pmatrix}
14 \\
36 \\
98 \\
\end{pmatrix}.
$$

We have:

$$
X^\top X \beta = X^\top y.
$$

Explicitly:

$$
\begin{bmatrix}
3 & 6 & 14 \\
6 & 14 & 36 \\
14 & 36 & 98 \\
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\beta_2 \\
\end{bmatrix}
\,=
\begin{bmatrix}
14 \\
36 \\
98 \\
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
0 \\
0 \\
1 \\
\end{bmatrix}.
$$

This reproduces exactly $y = 0 + 0\cdot x + 1\cdot x^2 = x^2$.

The solution here is unique since $X^\top X$ is invertible (its determinant exists since $X$ has full column rank).

---

## Multivariate Polynomial Regression

Single‑feature polynomial regression generalizes by replacing the scalar power basis with a chosen set of multivariate basis functions (monomials or other basis) $\phi_j(\mathbf{x})$. For $p$ input features and maximum total degree $d$, the monomial basis contains all terms of the form

$$
\prod_{r=1}^p x_r^{\alpha_r},\qquad \text{with } \sum_{r=1}^p \alpha_r \le d, \alpha_r \in \mathbb{N}.
$$

The number of such terms (including the constant) is

$$
N(p,d) = {p + d \choose d},
$$

This rapidly blows up the model's complexity.

But the estimator matrix still has the form:

$$
\beta = (X^\top X)^{-1} X^\top y.
$$

Practical notes:

- **Interactions**: multivariate polynomials include interaction terms (e.g. $x_1 x_2$). Decide whether to include all cross terms or restrict to separable powers.
- **Feature explosion**: with moderate $p$ and $d$ the number of columns becomes large. Use feature selection, orthogonal bases, or sparse methods.
- **Regularization** is often necessary in multivariate cases to control variance. Ridge or Elastic Net are common.

---

## When to Use

- When the relationship is smooth and globally well approximated by a low‑degree polynomial.
- When interpretability of polynomial coefficients is useful.

Avoid when:

- The relationship is locally complex (use splines or kernel methods).
- Dimensionality is high and interactions are many (use parsimonious models or regularized learners).

---

## Setup

### Prerequisites

- A C++ compiler: g++ (Windows), clang (macOS), or gcc (Linux).
- Terminal or command line interface.
- [Eigen library](https://gitlab.com/libeigen/eigen/-/releases) for matrix operations. Download, unzip, and include it in your project or in a system include path (e.g., `/usr/include/eigen3` on Linux).

### Installation

Clone the repository:

```bash
git clone https://github.com/intelligent-username/Polynomial-Regression.git
```

Run the `.exe` file in the main directory.

```bash
g++ -std=c++17 -I /path/to/eigen PolynomialRegression.cpp -o PR # Or whatever compiler you use. Note that adding the path to eigen is unnecessary if it's in a system include path.
```

```bash
./PR.exe
```

or import the `PolynomialRegression.h` file into your C++ project as a module.

---

## License

This project is under the MIT License. Use and adapt freely with attribution.
