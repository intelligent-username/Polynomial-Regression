// Basic 1D polynomial regression usage
#include "polynomial/PolynomialRegression.h"
#include <iostream>
#include <vector>

int main() {
    poly::PolynomialRegression model; // or just PolynomialRegression due to alias

    // Fit y = 2x + 1
    std::vector<double> x = {0,1,2,3};
    std::vector<double> y = {1,3,5,7};
    auto coeffs = model.fit(x, y, 1);

    std::cout << "Fitted linear coefficients (intercept, slope): ";
    for (double c : coeffs) std::cout << c << ' ';
    std::cout << "\nPredict x=4 -> " << model.predict1D(4.0, coeffs) << '\n';

    // Quadratic demo y = x^2
    std::vector<double> flat = {0,0, 1,1, 2,4, 3,9};
    auto quad = model.fit(flat, 2);
    std::cout << "Quadratic coefficients (c0,c1,c2): ";
    for (double c : quad) std::cout << c << ' ';
    std::cout << "\nPredict x=5 -> " << model.predict1D(5.0, quad) << '\n';

    return 0;
}
