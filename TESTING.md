# Testing Spine (Planned)

This file documents the intended testing approach. At user request, only the *framework* (spine) is present — no substantive test assertions yet.

## Goals
- Provide a place to add unit tests for polynomial regression behavior.
- Keep early scope small while mathematical foundations are still being learned.
- Allow future expansion (noise robustness, regularization, performance).

## Current Skeleton
```
./tests/test_spine.cpp
```
Contains:
- Basic program returning 0.
- Commented example of how a test might look.

## Planned Test Categories
1. Deterministic fits
   - Exact reconstruction of low‑degree polynomials (constant, linear, quadratic, cubic).
2. Overdetermined systems
   - Least squares behavior with extra points + tolerance checks.
3. Edge cases
   - Degree 0 input validation.
   - Degree >= number of points (expect exception).
   - Duplicate x values with varying y (still solvable in LS sense if enough points).
4. Numerical stability exploration (later)
   - Large magnitude x values vs. scaled x.
5. Future: Regularization
   - Ridge path sanity when added.

## Manual Test Ideas (for now)
Run the demo executable and visually inspect coefficients:
```
n=3 points for y=x^2      -> expect ~[0,0,1]
n=3 points for y=2x+1     -> expect ~[1,2]
```

## Guidelines (Once Implemented)
- Use a lightweight assertion helper or adopt a framework (e.g., doctest, Catch2) when ready.
- Keep tolerances explicit: `|pred - true| < 1e-9` for clean synthetic data.
- Separate data generation helpers (e.g., `generatePolynomialPoints(coeffs, xs)`).

## Extensibility Notes
- `CFile` wrapper can support fixture loading (e.g., serialized datasets) later.
- Could add a `bench/` directory for timing larger synthetic fits.

---
*This scaffold keeps momentum focused on learning while preparing for structured validation later.*
