// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file real_type.h
 * \brief Define macros to switch single/double precision.
 *
 */
#ifndef KALYPSSO_CORE_BINOMIAL_COEF_H_
#define KALYPSSO_CORE_BINOMIAL_COEF_H_

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>

#include <cstdint>

namespace kalypsso
{

/**
 * Compute binomial coefficient.
 */
KOKKOS_INLINE_FUNCTION
constexpr int
binomial_coef(int n, int k)
{
  int res = 1;
  if (k > n - k)
    k = n - k;
  for (int i = 0; i < k; ++i)
  {
    res *= (n - i);
    res /= (i + 1);
  }

  return res;

} // binomial_coef

/**
 * Provide a constexpr storage for binomial coefficients of the Pascal triangle from N=1 up to Nmax.
 */
template <int Nmax = 10>
class BinomialCoef
{
public:
  constexpr KOKKOS_INLINE_FUNCTION
  BinomialCoef()
    : m_pascal_triangle()
  {
    int i = 0;
    for (int N = 1; N <= Nmax; ++N)
    {
      for (int k = 0; k <= N; ++k)
      {
        m_pascal_triangle[i] = binomial_coef(N, k);
        ++i;
      }
    }
  }

  constexpr KOKKOS_INLINE_FUNCTION int
  num_coef() const
  {
    return static_cast<int>(NUM_COEFS);
  }

  constexpr KOKKOS_INLINE_FUNCTION int
  operator()(int N, int k) const
  {
    // static_assert(N <= Nmax, "N is too large");
    // static_assert(k >= 0 and k <= N, "k is invalid");

    return m_pascal_triangle[(N + 2) * (N - 1) / 2 + k];
  }

private:
  static constexpr size_t NUM_COEFS = (3 + Nmax + 1) * (Nmax - 1) / 2;
  int                     m_pascal_triangle[NUM_COEFS];
}; // class BinomialCoef

} // namespace kalypsso

#endif // KALYPSSO_CORE_BINOMIAL_COEF_H_
