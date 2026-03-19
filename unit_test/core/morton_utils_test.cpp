// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kalypsso/core/morton_utils.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_shared_test, morton_utils)
{

  uint32_t value = 2 + 4; // 0x10  + 0x100
  auto     v1 = kalypsso::splitBy3<3>(value);
  EXPECT_EQ(v1, 72) << "kalypsso::splitBy3<3> failed";

  value = 20;
  v1 = kalypsso::splitBy3<3>(value);
  EXPECT_EQ(v1, 4160) << "kalypsso::splitBy3<3> failed";

} // morton_utils

} // namespace kalypsso
