// Show Multi-feature polynomial regression (total degree basis)
#include "polynomial/PolynomialRegression.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    poly::PolynomialRegression model;

    // Target: y = 1 + 2*x + 3*z  (two features: x, z)
    std::vector<std::vector<double>> samples = {
        {0,0,1}, // y=1
        {1,0,3}, // 1+2*1
        {0,1,4}, // 1+3*1
        {1,1,6}, // 1+2+3
        {2,0,5}, // 1+4
        {0,2,7}  // 1+6
    };

    auto coeffs = model.fitMulti(samples, 1);
    std::cout << "Coefficients (expected ~[1, 2, 3]):\n";
    for (size_t i=0;i<coeffs.size();++i) std::cout << std::fixed << std::setprecision(6) << coeffs[i] << (i+1<coeffs.size()?' ':'\n');

    return 0;
}
