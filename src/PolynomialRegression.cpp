// src/PolynomialRegression.cpp (refactored for clarity & modularity)
#include "polynomial/PolynomialRegression.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace poly {

using Eigen::MatrixXd; using Eigen::VectorXd;

// ------------ Validation helpers ------------
namespace {
    void validate1DInput(const std::vector<double>& pts, int degree) {
        if (degree < 0) throw std::invalid_argument("Degree must be non-negative");
        if (pts.size() % 2 != 0) throw std::invalid_argument("Points vector length must be even (x,y pairs)");
        std::size_t n = pts.size()/2;
        if (n == 0) throw std::invalid_argument("At least one (x,y) point required");
        if (static_cast<std::size_t>(degree) >= n) {
            std::ostringstream oss; oss << "Need more data points (" << n << ") than degree (" << degree << ")";
            throw std::invalid_argument(oss.str());
        }
    }

    std::size_t validateMultiInput(const std::vector<std::vector<double>>& samples, int degree) {
        if (degree < 0) throw std::invalid_argument("Degree must be non-negative");
        if (samples.empty()) throw std::invalid_argument("At least one sample required");
        const auto& first = samples.front();
        if (first.size() < 2) throw std::invalid_argument("Each sample needs >=1 feature + target");
        std::size_t featureCount = first.size()-1;
        for (const auto& row : samples) if (row.size() != featureCount+1) throw std::invalid_argument("Inconsistent sample dimensionality");
        return featureCount;
    }
}

// ------------ Exponent enumeration (multi-feature) ------------
namespace {
    void enumerateExponentsRec(int featureCount, int pos, int remaining, std::vector<int>& current, std::vector<std::vector<int>>& out) {
        if (pos == featureCount) { out.push_back(current); return; }
        for (int k=0;k<=remaining;++k){ current[pos]=k; enumerateExponentsRec(featureCount,pos+1,remaining-k,current,out); }
    }
    std::vector<std::vector<int>> buildExponentList(int featureCount, int degree) {
        std::vector<std::vector<int>> exps; exps.reserve(128);
        std::vector<int> cur(featureCount,0);
        for (int total=0; total<=degree; ++total) enumerateExponentsRec(featureCount,0,total,cur,exps);
        return exps; // ordering mirrors original logic
    }
}

// ------------ Linear system solve helper ------------
namespace {
    std::vector<double> solveNormal(const MatrixXd& X, const VectorXd& y, const char* ctx) {
        MatrixXd XtX = X.transpose()*X;
        VectorXd Xty = X.transpose()*y;
        Eigen::LDLT<MatrixXd> ldlt(XtX);
        if (ldlt.info() != Eigen::Success) throw std::runtime_error(std::string("LDLT decomposition failed ")+ctx);
        VectorXd coeffs = ldlt.solve(Xty);
        if (ldlt.info() != Eigen::Success) throw std::runtime_error(std::string("Solve failed ")+ctx);
        return {coeffs.data(), coeffs.data()+coeffs.size()};
    }
}

// ------------ Design matrix builders ------------
namespace {
    void build1DDesign(const std::vector<double>& pts, int degree, MatrixXd& X, VectorXd& y) {
        std::size_t n = pts.size()/2; X.resize(static_cast<Eigen::Index>(n), degree+1); y.resize(static_cast<Eigen::Index>(n));
        for (std::size_t i=0;i<n;++i){
            double xi=pts[2*i]; double yi=pts[2*i+1]; y(static_cast<Eigen::Index>(i))=yi; double power=1.0;
            for (int j=0;j<=degree;++j){ X(static_cast<Eigen::Index>(i), j)=power; power*=xi; }
        }
    }
    void buildMultiDesign(const std::vector<std::vector<double>>& samples,
                          const std::vector<std::vector<int>>& exps,
                          MatrixXd& X, VectorXd& y) {
        std::size_t m = samples.size(); std::size_t termCount = exps.size();
        X.resize(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(termCount));
        y.resize(static_cast<Eigen::Index>(m));
        std::size_t featureCount = samples.front().size()-1;
        for (std::size_t i=0;i<m;++i){
            const auto& row = samples[i]; y(static_cast<Eigen::Index>(i))=row.back();
            for (std::size_t t=0;t<termCount;++t){
                double prod=1.0; const auto& expo = exps[t];
                for (std::size_t f=0; f<featureCount; ++f){ int e = expo[f]; if (e==0) continue; prod*= std::pow(row[f], e); }
                X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(t))=prod;
            }
        }
    }
}

// ------------ Public API Stuff -----------
std::vector<double> PolynomialRegression::fit(const std::vector<double>& points, int desiredDegree) {
    validate1DInput(points, desiredDegree);
    MatrixXd X; VectorXd y; build1DDesign(points, desiredDegree, X, y);
    return solveNormal(X, y, "(1D)");
}

std::vector<double> PolynomialRegression::fitMulti(const std::vector<std::vector<double>>& samples, int degree) {
    std::size_t featureCount = validateMultiInput(samples, degree); (void)featureCount; // featureCount unused beyond building design
    auto exps = buildExponentList(static_cast<int>(featureCount), degree);
    MatrixXd X; VectorXd y; buildMultiDesign(samples, exps, X, y);
    return solveNormal(X, y, "(multi)");
}

} // namespace poly

