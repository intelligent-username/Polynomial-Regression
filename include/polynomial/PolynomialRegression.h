// include/polynomial/PolynomialRegression.h
// Public interface for polynomial regression (supports 1D & multi-feature)
// Now wrapped in namespace poly. A global using-alias preserves previous
// code that referenced ::PolynomialRegression directly.

#ifndef POLYNOMIALREGRESSION_H
#define POLYNOMIALREGRESSION_H

#include <vector>
#include <utility>

namespace poly {

class PolynomialRegression {
public:
    // Core 1D fit polynomial of given degree.
    // Input format: flat vector {x0, y0, x1, y1, ...}
    // Returns coefficients for f(x) = sum_{k=0..d} beta_k x^k
    std::vector<double> fit(const std::vector<double>& points, int desiredDegree);

    // Multidimensional polynomial regression (points+target bundled per row).
    std::vector<double> fitMulti(const std::vector<std::vector<double>>& samples, int degree);

    // Overload: separate x & y arrays (wrapper builds flat vector)
    std::vector<double> fit(const std::vector<double>& x,
                            const std::vector<double>& y,
                            int degree) {
        if (x.size() != y.size()) throw std::runtime_error("x/y size mismatch");
        std::vector<double> flat; flat.reserve(x.size()*2);
        for (size_t i=0;i<x.size();++i){ flat.push_back(x[i]); flat.push_back(y[i]); }
        return fit(flat, degree);
    }

    // Overload: vector of pairs (wrapper builds flat vector)
    std::vector<double> fit(const std::vector<std::pair<double,double>>& xy,
                            int degree) {
        std::vector<double> flat; flat.reserve(xy.size()*2);
        for (auto &p : xy){ flat.push_back(p.first); flat.push_back(p.second); }
        return fit(flat, degree);
    }

    // Prediction (1D) utilities (do not validate size aggressively for speed)
    double predict1D(double x, const std::vector<double>& coeffs) const noexcept {
        double acc = 0.0; double p = 1.0;
        for (double c : coeffs) { acc += c * p; p *= x; }
        return acc;
    }
    std::vector<double> predict1D(const std::vector<double>& xs,
                                   const std::vector<double>& coeffs) const {
        std::vector<double> out; out.reserve(xs.size());
        for (double x : xs) out.push_back(predict1D(x, coeffs));
        return out;
    }
};

} // namespace poly

// Backward compatibility: allow existing code referencing ::PolynomialRegression
using PolynomialRegression = poly::PolynomialRegression;

#endif // POLYNOMIALREGRESSION_H

