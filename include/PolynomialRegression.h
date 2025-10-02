// include/PolynomialRegression.h
// Public interface for polynomial regression (educational, minimal)

#ifndef POLYNOMIALREGRESSION_H
#define POLYNOMIALREGRESSION_H

#include <vector>

class PolynomialRegression {
public:
    // Fit polynomial of given degree to points.
    // Input: flat vector of points: {x0, y0, x1, y1, ...}
    // degree: desired polynomial degree
    // Output: vector of coefficients [β0, β1, ..., βd]
    std::vector<double> fit(const std::vector<double>& points, int desiredDegree);
};

#endif // POLYNOMIALREGRESSION_H
