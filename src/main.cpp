// Quick little demo of P.R.

#include "polynomial/PolynomialRegression.h"
#include <iostream>
#include <vector>
#include <cstdio>

static void printCoeffs(const std::vector<double>& coeffs) {
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        std::cout << coeffs[i];
        if (i + 1 < coeffs.size()) std::cout << ' ';
    }
}

int main() {
    std::printf("Polynomial Regression from Scratch\n");

    PolynomialRegression model;

    {
        std::vector<double> flatPoints = {1,1, 2,4, 3,9};
        auto coeffs = model.fit(flatPoints, 2);
        std::cout << "Example 1 (y=x^2) coefficients: ";
        printCoeffs(coeffs);
        std::cout << "\n";
    }

    {
        std::vector<double> flatPoints = {0,1, 1,3, 2,5};
        auto coeffs = model.fit(flatPoints, 1);
        std::cout << "Example 2 (y=2x+1) coefficients: ";
        printCoeffs(coeffs);
        std::cout << "\n";
    }

    return 0;
}
