// SPDX-FileCopyrightText: 2025 kalypsso-core authors
// © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/mesh_utils.h>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_shared_test, mesh_utils_faces)
{

  // 2d
  {
    const auto dir0 = IY;
    const auto faces = Face::get_pair_of_faces<2>(dir0);

    EXPECT_EQ(faces[0] == Face::XMIN, false);
    EXPECT_EQ(faces[0] == Face::YMIN, true);
    EXPECT_EQ(faces[1] == Face::YMAX, true);
  }

  // 3d
  {
    const auto dir0 = IZ;
    const auto faces = Face::get_pair_of_faces<3>(dir0);

    EXPECT_EQ(faces[0] == Face::XMIN, false);
    EXPECT_EQ(faces[0] == Face::ZMIN, true);
    EXPECT_EQ(faces[1] == Face::YMAX, false);
  }


} // kalypsso_shared_test, mesh_utils_faces

} // namespace kalypsso
