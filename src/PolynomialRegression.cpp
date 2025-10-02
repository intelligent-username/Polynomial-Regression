// src/PolynomialRegression.cpp
#include "PolynomialRegression.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <sstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::vector<double> PolynomialRegression::fit(const std::vector<double>& points, int desiredDegree) {
    if (desiredDegree < 0) {
        throw std::invalid_argument("Degree must be non‑negative");
    }
    if (points.size() % 2 != 0) {
        throw std::invalid_argument("Points vector length must be even (x,y pairs)");
    }
    const std::size_t n = points.size() / 2;
    if (n == 0) {
        throw std::invalid_argument("At least one (x,y) point required");
    }
    if (static_cast<std::size_t>(desiredDegree) >= n) {
        std::ostringstream oss;
        oss << "Need strictly more data points (" << n
            << ") than polynomial degree (" << desiredDegree
            << ") for a unique least squares solution.";
        throw std::invalid_argument(oss.str());
    }

    MatrixXd X(n, desiredDegree + 1);
    VectorXd y(n);

    for (std::size_t i = 0; i < n; ++i) {
        double xi = points[2 * i];
        double yi = points[2 * i + 1];
        y(static_cast<Eigen::Index>(i)) = yi;
        double power = 1.0;
        for (int j = 0; j <= desiredDegree; ++j) {
            X(static_cast<Eigen::Index>(i), j) = power;
            power *= xi;
        }
    }

    MatrixXd XtX = X.transpose() * X;
    VectorXd Xty = X.transpose() * y;

    Eigen::LDLT<MatrixXd> ldlt(XtX);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT decomposition failed; matrix may be singular");
    }
    VectorXd coeffs = ldlt.solve(Xty);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve normal equations");
    }

    return std::vector<double>(coeffs.data(), coeffs.data() + coeffs.size());
}
