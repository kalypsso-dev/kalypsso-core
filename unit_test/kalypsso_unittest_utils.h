// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KALYPSSO_UNIT_TEST_UTILS_H_
#define KALYPSSO_UNIT_TEST_UTILS_H_

#include "gtest/gtest.h"

#define EXPECT_REAL_EQ(val1, val2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<real_t>, val1, val2)

#endif // KALYPSSO_UNIT_TEST_UTILS_H_
