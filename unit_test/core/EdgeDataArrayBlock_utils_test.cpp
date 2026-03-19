// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/EdgeDataArrayBlock_utils.h>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(EdgeDataArrayBlock_utils, test_get_block_size)
{
  // 2d
  {
    const auto ebSize = get_edge_block_size<2>({ 3, 5 }, IX);
    EXPECT_EQ(ebSize[IX] == 3, true);
    EXPECT_EQ(ebSize[IY] == 6, true);

    const auto ebSize2 = get_edge_block_size<2>({ 3, 5 }, IY);
    EXPECT_EQ(ebSize2[IX] == 4, true);
    EXPECT_EQ(ebSize2[IY] == 5, true);

    const auto ebSize3 = get_edge_block_size<2>({ 3, 5 }, IZ);
    EXPECT_EQ(ebSize3[IX] == 4, true);
    EXPECT_EQ(ebSize3[IY] == 6, true);
  }

  // 3d
  {
    const auto ebSize = get_edge_block_size<3>({ 3, 5, 4 }, IX);
    EXPECT_EQ(ebSize[IX] == 3, true);
    EXPECT_EQ(ebSize[IY] == 6, true);
    EXPECT_EQ(ebSize[IZ] == 5, true);

    const auto ebSize2 = get_edge_block_size<3>({ 3, 5, 4 }, IZ);
    EXPECT_EQ(ebSize2[IX] == 4, true);
    EXPECT_EQ(ebSize2[IY] == 6, true);
    EXPECT_EQ(ebSize2[IZ] == 4, true);
  }
}

TEST(EdgeDataArrayBlock_utils, test_edge_flat_index_unravel_emf)
{
  // 2d
  {
    block_size_t<2> bSize{ 3, 5 };

    const auto offsets = compute_edge_flat_index_offsets_emf<2>(bSize);
    EXPECT_EQ(offsets[0] == 0, true);
    EXPECT_EQ(offsets[1] == 0, true);
    EXPECT_EQ(offsets[2] == 0, true);
    EXPECT_EQ(offsets[3] == 24, true);


    // edge_multiindex_t ijk{ 2, 3, IZ };
    auto ijk = edge_flat_index_unravel_emf<2>(14, bSize, offsets);
    EXPECT_EQ(ijk[IX] == 2, true);
    EXPECT_EQ(ijk[IY] == 3, true);
  }

  // 3d
  {
    block_size_t<3> bSize{ 3, 5, 4 };

    const auto offsets = compute_edge_flat_index_offsets_emf<3>(bSize);
    EXPECT_EQ(offsets[0] == 0, true);
    EXPECT_EQ(offsets[1] == 90, true);
    EXPECT_EQ(offsets[2] == 190, true);
    EXPECT_EQ(offsets[3] == 286, true);

    // edge_multiindex_t ijk{ 2, 3, 1, IX };
    auto ijk = edge_flat_index_unravel_emf<3>(29, bSize, offsets);
    EXPECT_EQ(ijk[IX] == 2, true);
    EXPECT_EQ(ijk[IY] == 3, true);
    EXPECT_EQ(ijk[IZ] == 1, true);
    EXPECT_EQ(ijk[3] == IX, true);

    // edge_multiindex_t ijk2{ 1, 2, 1, IY };
    auto ijk2 = edge_flat_index_unravel_emf<3>(29 + 90, bSize, offsets);
    EXPECT_EQ(ijk2[IX] == 1, true);
    EXPECT_EQ(ijk2[IY] == 2, true);
    EXPECT_EQ(ijk2[IZ] == 1, true);
    EXPECT_EQ(ijk2[3] == IY, true);

    // edge_multiindex_t ijk2{ 1, 1, 1, IZ };
    auto ijk3 = edge_flat_index_unravel_emf<3>(29 + 190, bSize, offsets);
    EXPECT_EQ(ijk3[IX] == 1, true);
    EXPECT_EQ(ijk3[IY] == 1, true);
    EXPECT_EQ(ijk3[IZ] == 1, true);
    EXPECT_EQ(ijk3[3] == IZ, true);
  }
}
TEST(EdgeDataArrayBlock_utils, test_is_edge_at_block_surface)
{
  // 2d
  {
    block_size_t<2> bSize{ 4, 4 };

    EXPECT_EQ(is_edge_at_block_surface<2>({ 2, 2, IZ }, bSize, Face::XMIN), false);
    EXPECT_EQ(is_edge_at_block_surface<2>({ 0, 2, IZ }, bSize, Face::XMIN), true);
    EXPECT_EQ(is_edge_at_block_surface<2>({ 0, 2, IZ }, bSize, Face::XMIN), true);
    EXPECT_EQ(is_edge_at_block_surface<2>({ 0, 2, IZ }, bSize, Face::XMAX), false);
    EXPECT_EQ(is_edge_at_block_surface<2>({ 3, 3, IZ }, bSize, Face::XMAX), true);
    EXPECT_EQ(is_edge_at_block_surface<2>({ 3, 3, IZ }, bSize, Face::YMAX), true);
  }

  // 3d
  {
    block_size_t<3> bSize{ 5, 5, 5 };

    EXPECT_EQ(is_edge_at_block_surface<3>({ 2, 2, 4, IZ }, bSize, Face::ZMAX), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 0, 3, IZ }, bSize, Face::ZMAX), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 0, 3, IZ }, bSize, Face::XMIN), true);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 0, 3, IZ }, bSize, Face::YMIN), true);

    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 0, 3, IX }, bSize, Face::XMIN), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 0, 3, IX }, bSize, Face::YMIN), true);

    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 2, 3, IX }, bSize, Face::XMIN), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 0, 2, 3, IX }, bSize, Face::YMIN), false);

    EXPECT_EQ(is_edge_at_block_surface<3>({ 4, 0, 3, IY }, bSize, Face::XMIN), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 4, 0, 3, IY }, bSize, Face::YMIN), false);

    EXPECT_EQ(is_edge_at_block_surface<3>({ 2, 4, 1, IX }, bSize, Face::YMAX), true);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 2, 4, 1, IY }, bSize, Face::YMAX), false);
    EXPECT_EQ(is_edge_at_block_surface<3>({ 2, 4, 1, IZ }, bSize, Face::YMAX), true);
  }
}

TEST(EdgeDataArrayBlock_utils, test_is_edge_at_block_edge)
{
  // 2d
  {
    block_size_t<2> bSize{ 4, 4 };

    EXPECT_EQ(is_edge_at_block_edge<2>({ 2, 2, IZ }, bSize, Face::XMIN, Face::YMIN), false);

    EXPECT_EQ(is_edge_at_block_edge<2>({ 0, 0, IZ }, bSize, Face::XMIN, Face::YMIN), true);

    EXPECT_EQ(is_edge_at_block_edge<2>({ 0, 3, IZ }, bSize, Face::XMIN, Face::YMAX), true);

    EXPECT_EQ(is_edge_at_block_edge<2>({ 0, 3, IZ }, bSize, Face::XMAX, Face::YMAX), false);
  }

  // 3d
  {
    block_size_t<3> bSize{ 6, 6, 6 };

    EXPECT_EQ(is_edge_at_block_edge<3>({ 2, 2, 2, IZ }, bSize, Face::XMIN, Face::YMIN), false);

    EXPECT_EQ(is_edge_at_block_edge<3>({ 0, 0, 0, IZ }, bSize, Face::XMIN, Face::YMIN), true);

    EXPECT_EQ(is_edge_at_block_edge<3>({ 0, 0, 0, IX }, bSize, Face::XMIN, Face::ZMIN), false);
    EXPECT_EQ(is_edge_at_block_edge<3>({ 0, 0, 0, IX }, bSize, Face::YMIN, Face::ZMIN), true);

    EXPECT_EQ(is_edge_at_block_edge<3>({ 3, 0, 0, IX }, bSize, Face::YMIN, Face::ZMIN), true);
    EXPECT_EQ(is_edge_at_block_edge<3>({ 3, 5, 0, IX }, bSize, Face::YMAX, Face::ZMIN), true);

    EXPECT_EQ(is_edge_at_block_edge<3>({ 5, 5, 0, IY }, bSize, Face::XMAX, Face::ZMIN), true);
    EXPECT_EQ(is_edge_at_block_edge<3>({ 5, 5, 5, IY }, bSize, Face::XMAX, Face::ZMIN), false);
  }
}

TEST(EdgeDataArrayBlock_utils, test_get_edge_outside_unit_vector)
{

  // 2d
  {
    {
      const auto v = get_edge_outside_unit_vector<2>(Face::XMIN, Face::YMAX);
      EXPECT_EQ(v[0] == -1, true);
      EXPECT_EQ(v[1] == 1, true);
    }

    {
      const auto v = get_edge_outside_unit_vector<2>(Face::XMAX, Face::YMAX);
      EXPECT_EQ(v[0] == 1, true);
      EXPECT_EQ(v[1] == 1, true);
    }
  }

  // 3d
  {
    {
      const auto v = get_edge_outside_unit_vector<3>(Face::XMIN, Face::YMAX);
      EXPECT_EQ(v[0] == -1, true);
      EXPECT_EQ(v[1] == 1, true);
      EXPECT_EQ(v[2] == 0, true);
    }

    {
      const auto v = get_edge_outside_unit_vector<3>(Face::XMIN, Face::ZMAX);
      EXPECT_EQ(v[0] == -1, true);
      EXPECT_EQ(v[1] == 0, true);
      EXPECT_EQ(v[2] == 1, true);
    }
  }

} // EdgeDataArrayBlock_utils, test_get_edge_outside_unit_vector

} // namespace kalypsso
