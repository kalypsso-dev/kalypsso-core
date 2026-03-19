// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/DataArrayBlock_utils.h>

#include "gtest/gtest.h"

namespace kalypsso
{

// =================================================================
// =================================================================
TEST(DataArrayBlock_utils, test_get_number_of_cells_on_face)
{
  // 2d
  {
    block_size_t<2> bSize{ 3, 5 };

    EXPECT_EQ((get_number_of_cells_on_face<2, IX>(bSize) == 5), true);
    EXPECT_EQ((get_number_of_cells_on_face<2, IY>(bSize) == 3), true);
  }

  // 3d
  {
    block_size_t<3> bSize{ 7, 2, 5 };

    EXPECT_EQ((get_number_of_cells_on_face<3, IX>(bSize) == 10), true);
    EXPECT_EQ((get_number_of_cells_on_face<3, IY>(bSize) == 35), true);
    EXPECT_EQ((get_number_of_cells_on_face<3, IZ>(bSize) == 14), true);
  }
}

// =================================================================
// =================================================================
TEST(DataArrayBlock_utils, test_get_number_of_surface_cells)
{
  // 2d
  {
    block_size_t<2> bSize{ 3, 5 };

    EXPECT_EQ((get_number_of_surface_cells<2>(bSize) == 16), true);
  }

  // 3d
  {
    block_size_t<3> bSize{ 3, 5, 4 };

    EXPECT_EQ((get_number_of_surface_cells<3>(bSize) == 94), true);
  }
}

// =================================================================
// =================================================================
TEST(DataArrayBlock_utils, test_surface_flatindex_unravel)
{
  // 2d
  {
    block_size_t<2> bSize{ 4, 7 };

    {
      int32_t flat_index = 0;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 0, true);
      EXPECT_EQ(ij[IY] == 0, true);
      EXPECT_EQ(n[IX] == -1 and n[IY] == 0, true);
    }
    {
      int32_t flat_index = 5;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 0, true);
      EXPECT_EQ(ij[IY] == 5, true);
      EXPECT_EQ(n[IX] == -1 and n[IY] == 0, true);
    }
    {
      int32_t flat_index = 10;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 3, true);
      EXPECT_EQ(ij[IY] == 3, true);
      EXPECT_EQ(n[IX] == 1 and n[IY] == 0, true);
    }
    {
      int32_t flat_index = 15;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 1, true);
      EXPECT_EQ(ij[IY] == 0, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == -1, true);
    }
    {
      int32_t flat_index = 18;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 0, true);
      EXPECT_EQ(ij[IY] == 6, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 1, true);
    }
    {
      int32_t flat_index = 20;
      auto    ij = surface_flatindex_unravel_to_cell_ijk<2>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<2>(flat_index, bSize);

      EXPECT_EQ(ij[IX] == 2, true);
      EXPECT_EQ(ij[IY] == 6, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 1, true);
    }
  }

  // 3d
  {
    block_size_t<3> bSize{ 5, 4, 3 };

    {
      int32_t flat_index = 0;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 0, true);
      EXPECT_EQ(ijk[IY] == 0, true);
      EXPECT_EQ(ijk[IZ] == 0, true);
      EXPECT_EQ(n[IX] == -1 and n[IY] == 0 and n[IZ] == 0, true);
    }
    {
      int32_t flat_index = 18;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 2, true);
      EXPECT_EQ(ijk[IZ] == 1, true);
      EXPECT_EQ(n[IX] == 1 and n[IY] == 0 and n[IZ] == 0, true);
    }
    {
      int32_t flat_index = 28;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 0, true);
      EXPECT_EQ(ijk[IZ] == 0, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == -1 and n[IZ] == 0, true);
    }
    {
      int32_t flat_index = 38;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 0, true);
      EXPECT_EQ(ijk[IZ] == 2, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == -1 and n[IZ] == 0, true);
    }
    {
      int32_t flat_index = 48;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 3, true);
      EXPECT_EQ(ijk[IZ] == 1, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 1 and n[IZ] == 0, true);
    }
    {
      int32_t flat_index = 58;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 0, true);
      EXPECT_EQ(ijk[IZ] == 0, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 0 and n[IZ] == -1, true);
    }
    {
      int32_t flat_index = 68;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 2, true);
      EXPECT_EQ(ijk[IZ] == 0, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 0 and n[IZ] == -1, true);
    }
    {
      int32_t flat_index = 78;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 0, true);
      EXPECT_EQ(ijk[IZ] == 2, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 0 and n[IZ] == 1, true);
    }
    {
      int32_t flat_index = 88;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 2, true);
      EXPECT_EQ(ijk[IZ] == 2, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 0 and n[IZ] == 1, true);
    }
    {
      int32_t flat_index = 93;
      auto    ijk = surface_flatindex_unravel_to_cell_ijk<3>(flat_index, bSize);
      auto    n = surface_flatindex_to_normal_vector<3>(flat_index, bSize);

      EXPECT_EQ(ijk[IX] == 4, true);
      EXPECT_EQ(ijk[IY] == 3, true);
      EXPECT_EQ(ijk[IZ] == 2, true);
      EXPECT_EQ(n[IX] == 0 and n[IY] == 0 and n[IZ] == 1, true);
    }
  }
}

} // namespace kalypsso
