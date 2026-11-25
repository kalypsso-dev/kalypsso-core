// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborLocalOffset_test.cpp
 *
 * unit test for class NeighborLocalOffset.
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/NeighborLocalOffsets.h>

#include <cstdlib>

#include "../main_kalypsso_unittest.h"

#include "gtest/gtest.h"

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
TEST(kalypsso_core, NeighborLocalOffsets_decode_test)
{

  {

    constexpr size_t dim = 2;
    auto             local_offsets = NeighborLocalOffsets<dim>::local_offsets_t{};

    local_offsets[0] = 0xF0ULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0x0);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 1), 0xF);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 2), 0x0);
    local_offsets[0] = 0xBAULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0xA);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 1), 0xB);
    local_offsets[0] = 0xD125C38BAEULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 9), 0xD);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 8), 0x1);
  }

  {
    constexpr size_t dim = 3;
    auto             local_offsets = NeighborLocalOffsets<dim>::local_offsets_t{};

    local_offsets[0] = 0xF0ULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0x30);
    local_offsets[0] = 0xABCDEF;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 47);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 1), 55);

    local_offsets[1] = 0xDEADBEEFULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 9), 47);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 13), 30);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 12), 43);

    local_offsets[2] = 0x65234ADDEADBEEFULL;
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 27), 25);
  }
}

// ====================================================================
// ====================================================================
TEST(kalypsso_core, NeighborLocalOffsets_encode_test)
{

  {
    constexpr size_t dim = 2;
    auto             local_offsets = NeighborLocalOffsets<dim>::local_offsets_t{};

    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, 0xA);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 1, 0x2);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0xA);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 1), 0x2);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 2), 0x0);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, 0xB);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 1, 0xA);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0xB);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 1), 0xA);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, 0x7);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 1, 0xA);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 2, 0xB);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 3, 0x8);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 4, 0x3);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 5, 0xC);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 6, 0x5);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 7, 0x2);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 8, 0x1);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 9, 0x9);
    EXPECT_EQ(local_offsets[0], 0x9125C38BA7ULL);
  }

  {
    constexpr size_t dim = 3;
    auto             local_offsets = NeighborLocalOffsets<dim>::local_offsets_t{};

    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, 0x30);
    EXPECT_EQ(NeighborLocalOffsets<dim>::decode_local_offset(local_offsets, 0), 0x30);

    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 0, 47);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 1, 0x37);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 2, 0x2C);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 3, 0x1A);
    EXPECT_EQ(local_offsets[0], 0x6ACDEFULL);

    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 18, 0x2F);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 19, 0x38);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 20, 0x1B);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 21, 0x2B);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 22, 0x1E);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 23, 0x37);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 24, 0x0A);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 25, 0x0D);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 26, 0x12);
    NeighborLocalOffsets<dim>::encode_local_offset(local_offsets, 27, 0x19);
    EXPECT_EQ(local_offsets[2], 0x65234ADDEADBE2FULL);
  }
}

} // namespace core

} // namespace kalypsso
