
#include <iostream>
#include <vector>
#include <cmath>
#include "polynomial/PolynomialRegression.h"

static bool near(double a, double b, double eps=1e-9) { return std::fabs(a-b) < eps; }

int main() {
    poly::PolynomialRegression pr;
    // Target: y = 1 + 2*x + 3*z  (features: x,z)
    std::vector<std::vector<double>> samples = {
        {0,0,1}, {1,0,3}, {0,1,4}, {1,1,6}, {2,0,5}, {0,2,7}
    };
    auto coeffs = pr.fitMulti(samples, 1);
    if (coeffs.size() != 3 || !near(coeffs[0],1.0) || !near(coeffs[1],2.0) || !near(coeffs[2],3.0)) {
        std::cerr << "[MULTI] Coefficient check failed\n";
        return 1;
    }
    std::cout << "[MULTI] OK\n";
    return 0;
}
