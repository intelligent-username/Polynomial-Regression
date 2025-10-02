// tests/tests.cpp (original tests skeleton)
#include <iostream>
#include <vector>
#include "PolynomialRegression.h"

static void printVector(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); i++) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

int main() {
    std::cout << "==== PolynomialRegression: Basic Tests (Skeleton) ====" << std::endl;
    // Placeholder – migrate assertions here later.
    return 0;
}
#include <iostream>
#include <vector>
#include "PolynomialRegression.h"

using namespace std;

static void printVector(const vector<double>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i + 1 < v.size()) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "==== PolynomialRegression: Basic Tests (Spine Copy) ====" << endl;
    // Placeholder copy (original root tests.cpp retained for now until user confirms removal).
    // Future: consolidate into one test executable under tests/.
    return 0;
}
