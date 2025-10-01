#include "PolynomialRegression.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

class PR {
public:
    vector<double> fit(const vector<vector<double>>& points, int desiredDegree) {
        int n = points.size();
        MatrixXd X(n, desiredDegree + 1);
        VectorXd y(n);

        for (int i = 0; i < n; i++) {
            double xi = points[i][0];
            double yi = points[i][1];
            y(i) = yi;
            for (int j = 0; j <= desiredDegree; j++) {
                X(i, j) = pow(xi, j);
            }
        }

        VectorXd coeffs = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        return vector<double>(coeffs.data(), coeffs.data() + coeffs.size());
    }
};

int main() {
    printf("Polynomial Regression from Scratch\n");

    PR model;

    vector<vector<double>> points1 = {{1,1}, {2,4}, {3,9}};
    auto coeffs1 = model.fit(points1, 2);
    cout << "Test 1 coefficients: ";
    for (double c : coeffs1) cout << c << " ";
    cout << endl;

    vector<vector<double>> points2 = {{0,1}, {1,3}, {2,5}};
    auto coeffs2 = model.fit(points2, 1);
    cout << "Test 2 coefficients: ";
    for (double c : coeffs2) cout << c << " ";
    cout << endl;

    return 1;
}
