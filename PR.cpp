// BASIC POLYNOMIAL REGRESSION FROM SCRATCH
#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

class PolynomialRegression {
public:
    vector<double> fit(const vector<double>& points, int desiredDegree) {
        vector<double> coeffs(desiredDegree + 1, 0.0);
        return coeffs;
    }
};

void testEigen() {
    //  JUST TESTING EIGEN FOR NOW, will implement this tomorrow
    Matrix2d A;
    A << 1, 2,
         3, 4;
    Vector2d b(5, 6);
    Vector2d x = A.colPivHouseholderQr().solve(b);
    cout << "Matrix A:\n" << A << "\n\n";
    cout << "Vector b:\n" << b << "\n\n";
    cout << "Solution x:\n" << x << "\n";

    // TODO: regression logic
    // return coeffs;

}

int main() {
    cout << "Running Eigen test...\n";
    testEigen();
    return 0;
}
