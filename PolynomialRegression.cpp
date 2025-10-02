/**
 * PolynomialRegression.cpp
 * ------------------------------------------------------------
 * Implementation of a very small educational polynomial regression
 * utility using the normal equation ( (X^T X) beta = X^T y ).
 *
 * Design goals:
 *  - Keep the public interface minimal (single fit method for now)
 *  - Make future extension (predict, metrics, regularization) easy
 *  - Provide clear precondition checks with actionable error messages
 *  - Avoid premature abstraction while you are still learning concepts
 *
 * Notes:
 *  - Uses Eigen for linear algebra. LDLT decomposition is used for
 *    reasonable numerical stability over directly forming an explicit
 *    inverse. This keeps the code short while avoiding (X^T X)^{-1}.
 *  - Input format matches the existing header & tests: a flat vector
 *    {x0, y0, x1, y1, ...} to keep the CLI / examples trivial.
 *  - Throws std::invalid_argument on malformed input so callers can
 *    catch and react (e.g. in future test harness / examples).
 */

#include "PolynomialRegression.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <sstream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::vector<double> PolynomialRegression::fit(const std::vector<double>& points, int desiredDegree) {
    // ---- Basic validation -------------------------------------------------
    if (desiredDegree < 0) {
        throw std::invalid_argument("Degree must be non‑negative");
    }
    if (points.size() % 2 != 0) {
        throw std::invalid_argument("Points vector length must be even (x,y pairs)");
    }
    const std::size_t n = points.size() / 2; // number of (x,y) pairs
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

    // ---- Build design matrix X and target vector y -----------------------
    MatrixXd X(n, desiredDegree + 1);
    VectorXd y(n);

    for (std::size_t i = 0; i < n; ++i) {
        double xi = points[2 * i];
        double yi = points[2 * i + 1];
        y(static_cast<Eigen::Index>(i)) = yi;
        // x^0 ... x^d
        double power = 1.0; // xi^0
        for (int j = 0; j <= desiredDegree; ++j) {
            X(static_cast<Eigen::Index>(i), j) = power;
            power *= xi; // iterative multiply is faster & more stable than pow each time
        }
    }

    // ---- Solve normal equations using a decomposition --------------------
    // We avoid (X.transpose()*X).inverse() * X.transpose()*y directly.
    MatrixXd XtX = X.transpose() * X;
    VectorXd Xty = X.transpose() * y;

    // LDLT gives a fast solution for symmetric positive semidefinite matrices.
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
