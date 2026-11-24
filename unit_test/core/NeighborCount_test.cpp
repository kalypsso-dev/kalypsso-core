// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborCount_test.cpp
 *
 * unit test for class NeighborCount.
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/ComputeNumberOfNeighborsOfQuadrant.h>

#include <cstdlib>

#include "../main_kalypsso_unittest.h"

#include "gtest/gtest.h"

namespace kalypsso
{

// ====================================================================
// ====================================================================
TEST(kalypsso_core, NeighborCount_decode_test)
{

  {

    constexpr size_t dim = 2;
    auto             count = kalypsso::core::NeighborCount<dim>();

    count.local_offsets[0] = 0xF0ULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0x0);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 1), 0xF);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 2), 0x0);
    count.local_offsets[0] = 0xBAULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0xA);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 1), 0xB);
    count.local_offsets[0] = 0xD125C38BAEULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 9), 0xD);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 8), 0x1);
  }

  {
    constexpr size_t dim = 3;
    auto             count = kalypsso::core::NeighborCount<dim>();

    count.local_offsets[0] = 0xF0ULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0x30);
    count.local_offsets[0] = 0xABCDEF;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 47);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 1), 55);

    count.local_offsets[1] = 0xDEADBEEFULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 9), 47);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 13), 30);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 12), 43);

    count.local_offsets[2] = 0x65234ADDEADBEEFULL;
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 27), 25);
  }
}

// ====================================================================
// ====================================================================
TEST(kalypsso_core, NeighborCount_encode_test)
{

  {
    constexpr size_t dim = 2;
    auto             count = kalypsso::core::NeighborCount<dim>();

    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 0, 0xA);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 1, 0x2);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0xA);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 1), 0x2);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 2), 0x0);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 0, 0xB);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 1, 0xA);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0xB);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 1), 0xA);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 0, 0x7);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 1, 0xA);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 2, 0xB);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 3, 0x8);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 4, 0x3);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 5, 0xC);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 6, 0x5);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 7, 0x2);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 8, 0x1);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 9, 0x9);
    EXPECT_EQ(count.local_offsets[0], 0x9125C38BA7ULL);
  }

  {
    constexpr size_t dim = 3;
    auto             count = kalypsso::core::NeighborCount<dim>();

    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 0, 0x30);
    EXPECT_EQ(kalypsso::core::NeighborCount<dim>::decode_local_offset(count, 0), 0x30);

    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 0, 47);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 1, 0x37);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 2, 0x2C);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 3, 0x1A);
    EXPECT_EQ(count.local_offsets[0], 0x6ACDEFULL);

    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 18, 0x2F);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 19, 0x38);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 20, 0x1B);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 21, 0x2B);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 22, 0x1E);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 23, 0x37);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 24, 0x0A);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 25, 0x0D);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 26, 0x12);
    kalypsso::core::NeighborCount<dim>::encode_local_offset(count, 27, 0x19);
    EXPECT_EQ(count.local_offsets[2], 0x65234ADDEADBE2FULL);
  }
}

} // namespace kalypsso
