# osqp-cpp: A C++ wrapper for [OSQP](https://osqp.org/)

A C++ wrapper for [OSQP](https://github.com/oxfordcontrol/osqp), an
[ADMM](http://stanford.edu/~boyd/admm.html)-based solver for
[quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming).

Compared with OSQP's native C interface, the wrapper provides a more convenient
input format using Eigen sparse matrices and handles the lifetime of the
`OSQPWorkspace` struct. This package has similar functionality to
[osqp-eigen](https://github.com/robotology/osqp-eigen).

The full API is documented in-line in `osqp++.h`. We describe only the input
format in this README.

Note: OSQP uses looser default tolerances than other similar solvers. We
recommend looking at the description of the convergence tolerances in Section
3.4 of the OSQP [paper](https://arxiv.org/abs/1711.08013) and adjusting
tolerances via the `OsqpSettings` struct as appropriate.

This is not an officially supported Google product.

## `OsqpInstance` format

OSQP solves the convex quadratic optimization problem:

```
min_x 0.5 * x'Px + q'x
s.t.  l <= Ax <= u
```

where `P` is a symmetric positive semi-definite matrix.

The inequalities are component-wise, and equalities may be enforced by setting
`l[i] == u[i]` for some row `i`. Single-sided inequalities can be enforced by
setting the lower or upper bounds to negative or positive infinity
(`std::numeric_limits<double>::infinity()`), respectively.

This maps to the `OsqpInstance` struct in `osqp++.h` as follows.

-   `objective_matrix` is `P`.
-   `objective_vector` is `q`.
-   `constraint_matrix` is `A`.
-   `lower_bounds` is `l`.
-   `upper_bounds` is `u`.

## Example usage

The code below formulates and solves the following 2-dimensional optimization
problem:

```
min_(x,y) x^2 + 0.5 * x * y + y^2 + x
s.t.      x >= 1
```

```C++
const double kInfinity = std::numeric_limits<double>::infinity();
SparseMatrix<double> objective_matrix(2, 2);
const Triplet<double> kTripletsP[] = {
    {0, 0, 2.0}, {1, 0, 0.5}, {0, 1, 0.5}, {1, 1, 2.0}};
objective_matrix.setFromTriplets(std::begin(kTripletsP),
                                   std::end(kTripletsP));

SparseMatrix<double> constraint_matrix(1, 2);
const Triplet<double> kTripletsA[] = {{0, 0, 1.0}};
constraint_matrix.setFromTriplets(std::begin(kTripletsA),
                                      std::end(kTripletsA));

OsqpInstance instance;
instance.objective_matrix = objective_matrix;
instance.objective_vector.resize(2);
instance.objective_vector << 1.0, 0.0;
instance.constraint_matrix = constraint_matrix;
instance.lower_bounds.resize(1);
instance.lower_bounds << 1.0;
instance.upper_bounds.resize(1);
instance.upper_bounds << kInfinity;

OsqpSolver solver;
OsqpSettings settings;
// Edit settings if appropriate.
auto status = solver.Init(instance, settings);
// Assuming status.ok().
OsqpExitCode exit_code = solver.Solve();
// Assuming exit_code == OsqpExitCode::kOptimal.
double optimal_objective = solver.objective_value();
Eigen::VectorXd optimal_solution = solver.primal_solution();
```

## Installation (Unix)

osqp-cpp requires CMake, a C++17 compiler, and the following packages:

- [OSQP](https://github.com/oxfordcontrol/osqp) (compiled with 64-bit integers)
- [abseil-cpp](https://github.com/abseil/abseil-cpp)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [googletest](https://github.com/google/googletest) (for testing only)

On Debian/Ubuntu systems you may install Eigen via the `libeigen3-dev` package.

osqp-cpp will attempt to automatically detect if the necessary targets exist as
part of the same project. If the necessary `OSQP`, `abseil-cpp`, or `googletest`
targets are not found, osqp-cpp will attempt to download the sources from their
GitHub repositories through the use of CMake's `FetchContent` functionality. If
the `Eigen3` targets are not found, osqp-cpp will attempt to find Eigen3 as a
system package. To prevent osqp-cpp from unnecessarily downloading target
dependencies, please ensure that any target dependencies that are already
available are included before osqp-cpp.

To build osqp-cpp, run the following from the `osqp-cpp` directory:

```sh
$ mkdir build; cd build
$ cmake ..
$ make
$ make test
```

The interface is regularly tested only on Linux. Contributions to support and
automatically test additional platforms are welcome.

## Installation (Windows)

*These instructions are maintained by the community.*

Install prerequisite packages:

```sh
$ vcpkg install eigen3:x64-windows
$ vcpkg install abseil:x64-windows
$ vcpkg install gtest:x64-windows
```

Then, run the following from the `osqp-cpp` directory:

```sh
$ mkdir build; cd build
$ cmake ..
$ cmake --build .
$ cd Debug
```

## FAQ

-   Is OSQP deterministic?
    -   No, not in its default configuration. Section 5.2 of the OSQP
        [paper](https://arxiv.org/abs/1711.08013) describes that the update rule
        for the step size rho depends on the ratio between the runtime of the
        iterations and the runtime of the numerical factorization. Setting
        `adaptive_rho` to `false` disables this update rule and makes OSQP
        deterministic, but this could significantly slow down OSQP's
        convergence.
