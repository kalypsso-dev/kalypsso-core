// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <cstdint>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_shared_test, Kokkos_Array_extensions)
{

  Kokkos::Array<int32_t, 3> a1{ 12, 13, 14 };
  Kokkos::Array<int32_t, 3> a2{ 12, 13, 14 };
  Kokkos::Array<int32_t, 3> a3{ 42, 10, 20 };

  Kokkos::Array<int32_t, 3> a13p = a1 + a3;
  Kokkos::Array<int32_t, 3> a13m = a1 - a3;

  Kokkos::Array<int32_t, 3> a13p_true{ 54, 23, 34 };
  Kokkos::Array<int32_t, 3> a13m_true{ -30, 3, -6 };

  Kokkos::Array<int32_t, 3> a1x2{ 24, 26, 28 };
  Kokkos::Array<int32_t, 3> a1x3{ 36, 39, 42 };

  EXPECT_EQ(a1 == a2, true) << "test a1==a2 failed";
  EXPECT_EQ(a1 == a3, false) << "test a1!=a3 failed";

  EXPECT_EQ(a13p == a13p_true, true) << "test array addition failed";
  EXPECT_EQ(a13m == a13m_true, true) << "test array subtract failed";

  EXPECT_EQ(a1 * 2 == a1x2, true) << "test array multiplication by a scalar failed";
  EXPECT_EQ(3 * a1 == a1x3, true) << "test array multiplication by a scalar failed";


} // Kokkos_Array_extensions

} // namespace kalypsso
