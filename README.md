# Polynomial Regression From Scratch

![Cover](img.jpg)

---

## Outline

- [Polynomial Regression From Scratch](#polynomial-regression-from-scratch)
  - [Outline](#outline)
  - [Motivation \& Theory](#motivation--theory)
    - [The Optimization](#the-optimization)
  - [Installation](#installation)
  - [API](#api)
  - [License](#license)

---

## Motivation & Theory

Often times, we have a set of data points and want to generalize them to make useful predictions. This is largely what machine learning is about. Now, after engineering, cleaning, and transforming the data, you want to use it to train a model. However, the relationship between the features and the target variable may not always be linear.

Polynomial regression extends linear regression by adding extra predictors, obtained by raising each of the original predictors to a power. This allows the model to fit a wider range of curves and capture more complex relationships in the data.

Firstly, we have to define how we measure accuracy. The function that tells us how well our model is doing is called the **Loss Function**. Since this is our measure of accuracy, this is the metric we have to optimize for in order to create the ideal model. 

In this writeup, we will be using the Mean Squared Error (MSE) as our loss function, which is defined as:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

If we're modelling a polynomial of degree \(d\), our hypothesis function \(h(x)\) can be expressed as:

$$
h(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d
$$

Here:

- \(d\) is the degree of the polynomial.
- \(\beta_0, \beta_1, \ldots, \beta_d\) are the coefficients we need to learn from the data.
- \(x\) is the input feature.
- \(h(x)\) is the predicted output.

Now, notice that this is only working in two dimensions (i.e. one feature and one target variable). Expanding the scope to multiple features is straightforward, so I'll work with the single feature case for building the intuition and then proceed to generalize.

### The Optimization

## Installation

## API

## License
