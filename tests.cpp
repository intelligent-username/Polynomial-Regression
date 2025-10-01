#include <iostream>
#include <vector>
#include "PolynomialRegression.h"

using namespace std;

void printVector(const vector<double>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i + 1 < v.size()) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "==== PolynomialRegression: Basic Tests ====" << endl;
    PolynomialRegression pr;

    // Test 1: Constant function y = 2
    {
        vector<double> points = {0, 2, 1, 2, 2, 2};
        vector<double> coeffs = pr.fit(points, 0);
        cout << "Test 1 (constant y=2): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [2]" << endl;
    }

    // Test 2: Linear function y = 3x + 1
    {
        vector<double> points = {0, 1, 1, 4, 2, 7};
        vector<double> coeffs = pr.fit(points, 1);
        cout << "Test 2 (y=3x+1): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [1, 3]" << endl;
    }

    // Test 3: Quadratic function y = x^2
    {
        vector<double> points = {0, 0, 1, 1, 2, 4, 3, 9};
        vector<double> coeffs = pr.fit(points, 2);
        cout << "Test 3 (y=x^2): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [0, 0, 1]" << endl;
    }

    // Test 4: Quadratic with shift y = x^2 + 2x + 1
    {
        vector<double> points = {-1, 0, 0, 1, 1, 4, 2, 9};
        vector<double> coeffs = pr.fit(points, 2);
        cout << "Test 4 (y=x^2+2x+1): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [1, 2, 1]" << endl;
    }

    // Test 5: Cubic y = x^3
    {
        vector<double> points = {-2, -8, -1, -1, 0, 0, 1, 1, 2, 8};
        vector<double> coeffs = pr.fit(points, 3);
        cout << "Test 5 (y=x^3): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [0, 0, 0, 1]" << endl;
    }

    // Test 6: Overdetermined system (linear fit)
    {
        vector<double> points = {0, 1, 1, 2, 2, 3, 3, 4};
        vector<double> coeffs = pr.fit(points, 1);
        cout << "Test 6 (approx linear fit): coeffs = ";
        printVector(coeffs);
        cout << "  expected ≈ [1, 1]" << endl;
    }

    cout << "==== End of Tests ====" << endl;
    return 0;
}
