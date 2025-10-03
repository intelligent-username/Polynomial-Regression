// Basic 1D correctness checks (constant, linear, quadratic)
#include <iostream>
#include <vector>
#include <cmath>
#include "polynomial/PolynomialRegression.h"

static bool near(double a, double b, double eps=1e-9) { return std::fabs(a-b) < eps; }
static void printVec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i=0;i<v.size();++i){ std::cout<<v[i]; if(i+1<v.size()) std::cout<<","; }
    std::cout << "]";
}

int main() {
    std::cout << "[BASIC] PolynomialRegression tests" << std::endl;
    poly::PolynomialRegression pr;

    // Constant
    {
        std::vector<double> pts = {0,2, 1,2, 2,2};
        auto c = pr.fit(pts, 0);
        if (c.size()!=1 || !near(c[0],2.0)) { std::cerr << "Constant failed: "; printVec(c); std::cerr << "\n"; return 1; }
    }
    // Linear y=3x+1
    {
        std::vector<double> pts = {0,1, 1,4, 2,7};
        auto c = pr.fit(pts, 1);
        if (c.size()!=2 || !near(c[0],1.0) || !near(c[1],3.0)) { std::cerr << "Linear failed\n"; return 1; }
    }
    // Quadratic y=x^2
    {
        std::vector<double> pts = {0,0, 1,1, 2,4, 3,9};
        auto c = pr.fit(pts, 2);
        if (c.size()!=3 || !near(c[0],0.0) || !near(c[1],0.0) || !near(c[2],1.0)) { std::cerr << "Quadratic failed\n"; return 1; }
    }
    std::cout << "[BASIC] OK" << std::endl;
    return 0;
}
