// include/polynomial/PolynomialRegression.h
// Public interface for polynomial regression (supports 1D & multi-feature)

#ifndef POLYNOMIALREGRESSION_H
#define POLYNOMIALREGRESSION_H

#include <vector>

class PolynomialRegression {
public:
    // Fit polynomial of given degree to 1D data points.
    // Input format: flat vector {x0, y0, x1, y1, ...}
    // Returns coefficients for f(x) = sum_{k=0..d} beta_k x^k
    std::vector<double> fit(const std::vector<double>& points, int desiredDegree);

    // Multidimensional polynomial regression.
    // samples: each inner vector size = num_features + 1 (last element is y)
    // degree: maximum total degree of monomials included.
    // Monomials: all exponent tuples (e0..eF-1) with sum <= degree, lexicographic per increasing total and recursion order.
    std::vector<double> fitMulti(const std::vector<std::vector<double>>& samples, int degree);
};

#endif // POLYNOMIALREGRESSION_H
