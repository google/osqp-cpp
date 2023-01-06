// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OSQP_CPP_H_
#define OSQP_CPP_H_

// A C++ wrapper for OSQP (https://osqp.org/). See README.md for an overview.

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "Eigen/SparseCore"

namespace osqp {

// Must match the typedef in osqp/include/glob_opts.h (if not, it will trigger
// a static_assert failure in osqp++.cc).
// For consistency with ospq, this must be int on 32-bit systems, and
// long long on 64-bit systems.
#if INTPTR_MAX == INT64_MAX
using c_int = long long;  // NOLINT
#else
using c_int = int;  // NOLINT
#endif

// A memory-safe mirror of the OSQPData struct defined in osqp/include/types.h.
// The number of variables and constraints is implied by the shape of
// constraint_matrix. The format of the struct is further discussed in
// README.md. See also osqp++_test.cc for example usage.
struct OsqpInstance {
  c_int num_variables() const { return constraint_matrix.cols(); }
  c_int num_constraints() const { return constraint_matrix.rows(); }

  // Only the upper triangle of the objective matrix is read. The lower triangle
  // is ignored.
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> objective_matrix;
  Eigen::VectorXd objective_vector;
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> constraint_matrix;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
};

// This is a mirror of the OSQPSettings struct defined in
// osqp/include/types.h and documented at
// http://osqp.readthedocs.io/en/latest/interfaces/solver_settings.html. The
// names are unchanged and (hence) violate Google naming conventions. The
// default values are defined in osqp/include/constants.h. Note, OSQP's default
// settings are looser than other QP solvers. Do choose appropriate values of
// eps_abs and eps_rel for your application.
struct OsqpSettings {
  OsqpSettings();  // Sets default values.

  double rho;
  double sigma;
  c_int scaling;
  bool adaptive_rho;
  c_int adaptive_rho_interval;
  double adaptive_rho_tolerance;
  double adaptive_rho_fraction;
  c_int max_iter;
  double eps_abs;
  double eps_rel;
  double eps_prim_inf;
  double eps_dual_inf;
  double alpha;
  // linsys_solver is omitted. We don't change this.
  double delta;
  bool polish;
  c_int polish_refine_iter;
  bool verbose;
  bool scaled_termination;
  c_int check_termination;
  bool warm_start;
  double time_limit;
};

// Type-safe wrapper for OSQP's status codes that are defined at
// osqp/include/constants.h.
enum class OsqpExitCode {
  kOptimal,            // Optimal solution found.
  kPrimalInfeasible,   // Certificate of primal infeasibility found.
  kDualInfeasible,     // Certificate of dual infeasibility found.
  kOptimalInaccurate,  // Optimal solution found subject to reduced tolerances
  kPrimalInfeasibleInaccurate,  // Certificate of primal infeasibility found
                                // subject to reduced tolerances.
  kDualInfeasibleInaccurate,    // Certificate of dual infeasibility found
                                // subject to reduced tolerances.
  kMaxIterations,               // Maximum number of iterations reached.
  kInterrupted,                 // Interrupted by signal or CTRL-C.
  kTimeLimitReached,            // Ran out of time.
  kNonConvex,                   // The problem was found to be non-convex.
  kUnknown,                     // Unknown problem in solver.
};

std::string ToString(OsqpExitCode exitcode);

// This is a workaround to avoid including OSQP's header file. We can't directly
// forward-declare OSQPWorkspace because it is defined as a typedef of an
// anonymous struct.
struct OSQPWorkspaceHelper;

// This class is the main interface for calling OSQP. See example usage in
// README.md.
class OsqpSolver {
 public:
  OsqpSolver() = default;
  // Move-only.
  OsqpSolver(OsqpSolver&& rhs) = default;
  OsqpSolver& operator=(OsqpSolver&& rhs) = default;
  OsqpSolver(const OsqpSolver&) = delete;
  OsqpSolver& operator=(const OsqpSolver&) = delete;

  // Creates the internal OSQP workspace given the instance data and settings.
  // It is valid to call Init() multiple times.
  absl::Status Init(const OsqpInstance& instance, const OsqpSettings& settings);

  // Updates the elements of matrix the objective matrix P (upper triangular).
  // The new matrix should have the same sparsity structure.
  //
  // The solve will start from the previous optimal solution, which might not be
  // a good starting point given the new objective matrix. If that's the
  // case, one can call SetWarmStart with zero vectors to reset the state of the
  // solver.
  absl::Status UpdateObjectiveMatrix(
      const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>&
          objective_matrix);

  // Updates the elements of matrix the constraint matrix A.
  // The new matrix should have the same sparsity structure.
  absl::Status UpdateConstraintMatrix(
      const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>&
          constraint_matrix);

  // Combines call of UpdateObjectiveMatrix and UpdateConstraintMatrix.
  absl::Status UpdateObjectiveAndConstraintMatrices(
      const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>&
          objective_matrix,
      const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>&
          constraint_matrix);

  // Returns true if Init() has been called successfully.
  bool IsInitialized() const { return workspace_ != nullptr; }

  // Solves the instance by calling osqp_solve(). CHECK-fails if IsInitialized()
  // is false.
  ABSL_MUST_USE_RESULT OsqpExitCode Solve();

  // The number of iterations taken. CHECK-fails if IsInitialized() is false.
  c_int iterations() const;

  // The objective value of the primal solution. CHECK-fails if IsInitialized()
  // is false.
  double objective_value() const;

  // The primal solution, i.e., x. The Map is valid only for the lifetime of
  // the OSQP workspace. It will be invalidated by a call to Init() or if the
  // OsqpSolver is deleted. CHECK-fails if IsInitialized() is false.
  // Implementation details (do not depend on these): The underlying memory is
  // overwritten by SetPrimalWarmStart(). Modification of the problem data does
  // not destroy the solution.
  Eigen::Map<const Eigen::VectorXd> primal_solution() const;

  // The vector of lagrange multipliers on the linear constraints. The Map is
  // valid only for the lifetime of the OSQP workspace. It will be invalidated
  // by a call to Init() or if the OsqpSolver is deleted. CHECK-fails if
  // IsInitialized() is false. Implementation details (do not depend on these):
  // The underlying memory is overwritten by SetDualWarmStart(). Modification of
  // the problem data does not destroy the solution.
  Eigen::Map<const Eigen::VectorXd> dual_solution() const;

  // The primal infeasibility certificate. It is valid to query this only if
  // Solve() returns kPrimalInfeasible or kPrimalInfeasibleInaccurate. The
  // Map is valid only for the lifetime of the OSQP workspace. It will be
  // invalidated by a call to Init() or of the OsqpSolver is deleted.
  Eigen::Map<const Eigen::VectorXd> primal_infeasibility_certificate() const;

  // TODO(ml): Implement dual_infeasibility_certificate.

  // Sets a primal and dual warm-start for the next solve. Equivalent to
  // SetPrimalWarmStart(primal_vector) and SetDualWarmStart(dual_vector).
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if the vectors do not have expected dimensions
  // - UnknownError if the internal OSQP call fails
  // - OkStatus on success
  absl::Status SetWarmStart(
      const Eigen::Ref<const Eigen::VectorXd>& primal_vector,
      const Eigen::Ref<const Eigen::VectorXd>& dual_vector);

  // Sets a warm-start for the primal iterate for the next solve. Use a vector
  // of zeros to reset to the default initialization.
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if the vector does not have expected dimensions
  // - UnknownError if the internal OSQP call fails
  // - OkStatus on success
  absl::Status SetPrimalWarmStart(
      const Eigen::Ref<const Eigen::VectorXd>& primal_vector);

  // Sets a warm-start for the dual iterate for the next solve. Use a vector
  // of zeros to reset to the default initialization.
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if the vector does not have expected dimensions
  // - UnknownError if the internal OSQP call fails
  // - OkStatus on success
  absl::Status SetDualWarmStart(
      const Eigen::Ref<const Eigen::VectorXd>& dual_vector);

  // Sets the objective vector for the next solve. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if the vectors do not have expected dimensions
  // - UnknownError if the internal OSQP call fails
  // - OkStatus on success
  absl::Status SetObjectiveVector(
      const Eigen::Ref<const Eigen::VectorXd>& objective_vector);

  // Sets the lower_bounds and upper_bounds vectors for the next solve. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if the vectors do not have expected dimensions
  // - InvalidArgumentError if lower_bounds[i] > upper_bounds[i] for some i
  // - UnknownError if the internal OSQP call fails
  // - OkStatus on success
  absl::Status SetBounds(const Eigen::Ref<const Eigen::VectorXd>& lower_bounds,
                         const Eigen::Ref<const Eigen::VectorXd>& upper_bounds);

  // Gets the current value of the rho setting, i.e., the ADMM rho step. Returns
  // a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<double> GetRho() const;

  // Gets the current value of the sigma setting, i.e., the ADMM sigma step.
  // Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<double> GetSigma() const;

  // Gets the current value of the scaling setting, i.e., the number of
  // heuristic scaling iterations. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<c_int> GetScaling() const;

  // Gets the current value of the adaptive_rho setting, i.e., whether the rho
  // step size is adaptively set. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<bool> GetAdaptiveRho() const;

  // Gets the current value of the adaptive_rho_interval setting, i.e., the
  // number of iterations between rho adaptations. Returns a
  // FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<c_int> GetAdaptiveRhoInterval() const;

  // Gets the current value of the adaptive_rho_tolerance setting, i.e., the
  // tolerance X for adapting rho (the new value must be X times larger, or 1/X
  // times smaller, than the current value). Returns a FailedPreconditionError
  // if IsInitialized() is false.
  absl::StatusOr<double> GetAdaptiveRhoTolerance() const;

  // Gets the current value of the adaptive_rho_fraction setting, i.e., in
  // automatic mode (adaptive_rho_interval = 0), what fraction of setup time is
  // spent on selecting rho. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<double> GetAdaptiveRhoFraction() const;

  // Gets the current value of the max_iter setting, i.e., the maximum number of
  // iterations. Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<c_int> GetMaxIter() const;

  // Gets the current value of the eps_abs setting, i.e., the absolute error
  // tolerance for convergence. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<double> GetEpsAbs() const;

  // Gets the current value of the eps_rel setting, i.e., the relative error
  // tolerance for convergence. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<double> GetEpsRel() const;

  // Gets the current value of the eps_prim_inf setting, i.e., the absolute
  // error tolerance for primal infeasibility. Returns a FailedPreconditionError
  // if IsInitialized() is false.
  absl::StatusOr<double> GetEpsPrimInf() const;

  // Gets the current value of the eps_dual_inf setting, i.e., the absolute
  // error tolerance for dual infeasibility. Returns a FailedPreconditionError
  // if IsInitialized() is false.
  absl::StatusOr<double> GetEpsDualInf() const;

  // Gets the current value of the alpha setting, i.e., the ADMM overrelaxation
  // parameter. Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<double> GetAlpha() const;

  // Gets the current value of the delta setting, i.e., the polishing
  // regularization parameter. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<double> GetDelta() const;

  // Gets the current value of the polish setting, i.e., whether polishing is
  // performed. Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<bool> GetPolish() const;

  // Gets the current value of the polish_refine_iter setting, i.e., the number
  // of refinement iterations in polishing. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<c_int> GetPolishRefineIter() const;

  // Gets the current value of the verbose setting, i.e., whether solver output
  // is printed. Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<bool> GetVerbose() const;

  // Gets the current value of the scaled_termination setting, i.e., whether
  // scaled termination criteria is used. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<bool> GetScaledTermination() const;

  // Gets the current value of the check_termination setting, i.e., the interval
  // for checking termination. Returns a FailedPreconditionError if
  // IsInitialized() is false.
  absl::StatusOr<c_int> GetCheckTermination() const;

  // Gets the current value of the warm_start setting, i.e., if warm starting is
  // performed. Returns a FailedPreconditionError if IsInitialized() is false.
  absl::StatusOr<bool> GetWarmStart() const;

  // Gets the current value of the time_limit setting, i.e., the time limit as
  // expressed in seconds. Returns a FailedPreconditionError if IsInitialized()
  // is false.
  absl::StatusOr<double> GetTimeLimit() const;

  // Updates the rho setting, i.e., the ADMM rho step. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if rho_new <= 0.0
  // - OkStatus on success
  absl::Status UpdateRho(double rho_new);

  // Updates the max_iter setting, i.e., the maximum number of iterations.
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if max_iter_new <= 0
  // - OkStatus on success
  absl::Status UpdateMaxIter(int max_iter_new);

  // Updates the eps_abs setting, i.e., the absolute error tolerance for
  // convergence. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if eps_abs_new < 0.0
  // - OkStatus on success
  absl::Status UpdateEpsAbs(double eps_abs_new);

  // Updates the eps_rel setting, i.e., the relative error tolerance for
  // convergence. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if eps_rel_new < 0.0
  // - OkStatus on success
  absl::Status UpdateEpsRel(double eps_rel_new);

  // Updates the eps_prim_inf setting, i.e., the absolute error tolerance for
  // primal infeasibility. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if eps_prim_inf_new < 0.0
  // - OkStatus on success
  absl::Status UpdateEpsPrimInf(double eps_prim_inf_new);

  // Updates the eps_dual_inf setting, i.e., the absolute error tolerance for
  // dual infeasibility. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if eps_dual_inf_new < 0.0
  // - OkStatus on success
  absl::Status UpdateEpsDualInf(double eps_dual_inf_new);

  // Updates the alpha setting, i.e., the ADMM overrelaxation parameter.
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if !(0 < alpha_new < 2)
  // - OkStatus on success
  absl::Status UpdateAlpha(double alpha_new);

  // Updates the delta setting, i.e., the polishing regularization parameter.
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if delta_new <= 0.0
  // - OkStatus on success
  absl::Status UpdateDelta(double delta_new);

  // Updates the polish setting, i.e., whether polishing is performed. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - OkStatus on success
  absl::Status UpdatePolish(bool polish_new);

  // Updates the polish_refine_iter setting, i.e., the number of refinement
  // iterations in polishing. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if polish_refine_iter_new <= 0.0
  // - OkStatus on success
  absl::Status UpdatePolishRefineIter(int polish_refine_iter_new);

  // Updates the verbose setting, i.e., whether solver output is printed.
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - OkStatus on success
  absl::Status UpdateVerbose(bool verbose_new);

  // Updates the scaled_termination setting, i.e., whether scaled termination
  // criteria is used. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - OkStatus on success
  absl::Status UpdateScaledTermination(bool scaled_termination_new);

  // Updates the check_termination setting, i.e., the interval for checking
  // termination. Setting to zero disables termination checking. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if check_termination_new < 0.0
  // - OkStatus on success
  absl::Status UpdateCheckTermination(c_int check_termination_new);

  // Updates the warm_start setting, i.e., whether warm starting is performed.
  // Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - OkStatus on success
  absl::Status UpdateWarmStart(bool warm_start_new);

  // Updates the time_limit setting, i.e., the time limit as expressed in
  // seconds. Setting the time limit to zero disables time-limiting. Returns:
  // - FailedPreconditionError if IsInitialized() is false
  // - InvalidArgumentError if time_limit_new < 0.0
  // - OkStatus on success
  absl::Status UpdateTimeLimit(double time_limit_new);

 private:
  struct OsqpDeleter {
    void operator()(OSQPWorkspaceHelper* workspace) const;
  };

  std::unique_ptr<OSQPWorkspaceHelper, OsqpDeleter> workspace_;
};

}  // namespace osqp

#endif  // OSQP_CPP_H_
