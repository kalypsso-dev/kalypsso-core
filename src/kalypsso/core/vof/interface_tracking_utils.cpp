// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/vof/interface_tracking_utils.h>

namespace kalypsso
{

namespace vof
{

// ===============================================================================
// ===============================================================================
KOKKOS_FUNCTION real_t
compute_plane_rhs_pentagonal_newton(const real_t                   rhs,
                                    real_t                         guess,
                                    const Kokkos::Array<real_t, 3> normal)
{
  static constexpr int    MAX_ITERATIONS = KALYPSSO_YOUNGS_NEWTON_MAX_ITERATIONS;
  static constexpr real_t PRECISION = KALYPSSO_YOUNGS_NEWTON_PRECISION;
  const auto [nx, ny, _] = normal;

  real_t prev_guess;
  for (int i = 0; i < MAX_ITERATIONS; i++)
  {
    // clang-format off
    const real_t err = guess * guess * guess
                      - (guess - nx) * (guess - nx) * (guess - nx)
                      - (guess - ny) * (guess - ny) * (guess - ny)
                      - rhs;
    const real_t deriv = 3 * (guess * guess
                             - (guess - nx) * (guess - nx)
                             - (guess - ny) * (guess - ny));
    // clang-format on
    prev_guess = guess;
    guess = guess - err / deriv;

    if (abs(guess - prev_guess) < PRECISION)
      break;
  }

  return guess;

} // compute_plane_rhs_pentagonal_newton

// ===============================================================================
// ===============================================================================
KOKKOS_FUNCTION real_t
compute_plane_rhs_hexagonal_newton(const real_t                   rhs,
                                   real_t                         guess,
                                   const Kokkos::Array<real_t, 3> normal)
{
  static constexpr int    MAX_ITERATIONS = KALYPSSO_YOUNGS_NEWTON_MAX_ITERATIONS;
  static constexpr real_t PRECISION = KALYPSSO_YOUNGS_NEWTON_PRECISION;
  const auto [nx, ny, nz] = normal;

  real_t prev_guess;
  for (int i = 0; i < MAX_ITERATIONS; i++)
  {
    // clang-format off
    const real_t err = guess * guess * guess
                      - (guess - nx) * (guess - nx) * (guess - nx)
                      - (guess - ny) * (guess - ny) * (guess - ny)
                      - (guess - nz) * (guess - nz) * (guess - nz)
                      - rhs;
    const real_t deriv = 3 * (guess * guess
                             - (guess - nx) * (guess - nx)
                             - (guess - ny) * (guess - ny)
                             - (guess - nz) * (guess - nz));
    // clang-format on
    prev_guess = guess;
    guess = guess - err / deriv;

    if (abs(guess - prev_guess) < PRECISION)
      break;
  }

  return guess;

} // compute_plane_rhs_hexagonal_newton

} // namespace vof

} // namespace kalypsso
