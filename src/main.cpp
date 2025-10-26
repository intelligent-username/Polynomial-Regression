// Quick little demo of P.R.

#include "polynomial/PolynomialRegression.h"
#include <iostream>
#include <vector>
#include <cstdio>

using namespace std;

static void printCoeffs(const vector<double>& coeffs) {
    for (size_t i = 0; i < coeffs.size(); ++i) {
        cout << coeffs[i];
        if (i + 1 < coeffs.size()) cout << ' ';
    }
}

int main() {
    printf("Polynomial Regression from Scratch\n");

    PolynomialRegression model;

    {
        vector<double> flatPoints = {1,1, 2,4, 3,9};
        auto coeffs = model.fit(flatPoints, 2);
        cout << "Example 1 (y=x^2) coefficients: ";
        printCoeffs(coeffs);
        cout << "\n";
    }

    {
        vector<double> flatPoints = {0,1, 1,3, 2,5};
        auto coeffs = model.fit(flatPoints, 1);
        cout << "Example 2 (y=2x+1) coefficients: ";
        printCoeffs(coeffs);
        cout << "\n";
    }

    return 0;
}
