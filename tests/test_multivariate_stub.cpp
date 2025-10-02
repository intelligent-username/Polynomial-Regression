// tests/test_multivariate_stub.cpp
#include <iostream>
#include <vector>
#include "polynomial/PolynomialRegression.h"

int main() {
    PolynomialRegression pr;
    // y = 1 + 2*x1 + 3*x2
    std::vector<std::vector<double>> samples = {
        {0,0,1}, {1,0,3}, {0,1,4}, {1,1,6}, {2,0,5}, {0,2,7}
    };
    auto coeffs = pr.fitMulti(samples, 1);
    std::cout << "[MULTI] got " << coeffs.size() << " coefficients\n";
    return 0;
}
