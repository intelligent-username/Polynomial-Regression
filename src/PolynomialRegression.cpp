// src/PolynomialRegression.cpp
#include "polynomial/PolynomialRegression.h"
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

// Helper: recursively enumerate exponent tuples with total degree <= maxDegree.
static void enumerateExponents(int featureCount,
                               int maxDegree,
                               int pos,
                               int remaining,
                               std::vector<int>& current,
                               std::vector<std::vector<int>>& out) {
    if (pos == featureCount) {
        out.push_back(current);
        return;
    }
    for (int k = 0; k <= remaining; ++k) {
        current[pos] = k;
        enumerateExponents(featureCount, maxDegree, pos + 1, remaining - k, current, out);
    }
}

std::vector<double> PolynomialRegression::fitMulti(const std::vector<std::vector<double>>& samples, int degree) {
    if (degree < 0) throw std::invalid_argument("Degree must be non-negative");
    if (samples.empty()) throw std::invalid_argument("At least one sample required");
    std::size_t featureCount = 0;
    {
        const auto& first = samples.front();
        if (first.size() < 2) throw std::invalid_argument("Each sample must have at least 1 feature + target");
        featureCount = first.size() - 1;
    }
    for (const auto& row : samples) {
        if (row.size() != featureCount + 1) throw std::invalid_argument("Inconsistent sample dimensionality");
    }

    // Generate exponent tuples (monomials) with total degree <= degree (lexicographic by construction order).
    std::vector<std::vector<int>> exponents; exponents.reserve(128);
    std::vector<int> current(featureCount, 0);
    for (int total = 0; total <= degree; ++total) {
        enumerateExponents(static_cast<int>(featureCount), degree, 0, total, current, exponents);
    }

    const std::size_t m = samples.size();
    const std::size_t termCount = exponents.size();
    MatrixXd X(m, static_cast<Eigen::Index>(termCount));
    VectorXd y(m);

    for (std::size_t i = 0; i < m; ++i) {
        const auto& row = samples[i];
        y(static_cast<Eigen::Index>(i)) = row.back();
        for (std::size_t t = 0; t < termCount; ++t) {
            double prod = 1.0;
            const auto& expo = exponents[t];
            for (std::size_t f = 0; f < featureCount; ++f) {
                int e = expo[f];
                if (e == 0) continue;
                prod *= std::pow(row[f], e);
            }
            X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(t)) = prod;
        }
    }

    MatrixXd XtX = X.transpose() * X;
    VectorXd Xty = X.transpose() * y;
    Eigen::LDLT<MatrixXd> ldlt(XtX);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT decomposition failed (multi)");
    }
    VectorXd coeffs = ldlt.solve(Xty);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("Solve failed (multi)");
    }
    return std::vector<double>(coeffs.data(), coeffs.data() + coeffs.size());
}
