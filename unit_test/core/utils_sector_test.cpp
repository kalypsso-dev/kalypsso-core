// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_sector_test.cpp
 *
 * unit test for manipulating sectors.
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/utils_sector.h>
#include <cstdlib>

#include "../main_kalypsso_unittest.h"

#include "gtest/gtest.h"

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================

TEST(kalypsso_core, sector_2d)
{

  for (uint8_t sector = 0; sector < 9; ++sector)
  {
    if (is_face_sector_2d(sector))
    {
      const auto face = get_face_id_2d(sector);
      const auto face_str = face_to_string(face);
      printf("sector %d is a face   sector, face id  = %s\n", sector, face_str.c_str());
    }
    else if (is_corner_sector_2d(sector))
    {
      uint8_t face0, face1;
      get_corner_ids_2d(sector, face0, face1);
      const auto face0_str = face_to_string(face0);
      const auto face1_str = face_to_string(face1);
      printf("sector %d is a corner sector, face ids = %s,%s\n",
             sector,
             face0_str.c_str(),
             face1_str.c_str());
    }
  }
}

// ====================================================================
// ====================================================================

TEST(kalypsso_core, sector_3d)
{
  for (uint8_t sector = 0; sector < 27; ++sector)
  {
    if (is_face_sector_3d(sector))
    {
      const auto face = get_face_id_3d(sector);
      const auto face_str = face_to_string(face);
      printf("sector %d is a face   sector, face id  = %s\n", sector, face_str.c_str());
    }
    else if (is_edge_sector_3d(sector))
    {
      uint8_t face0, face1;
      get_edge_ids_3d(sector, face0, face1);
      const auto face0_str = face_to_string(face0);
      const auto face1_str = face_to_string(face1);
      printf("sector %d is a edge   sector, face ids = %s,%s\n",
             sector,
             face0_str.c_str(),
             face1_str.c_str());
    }
    else if (is_corner_sector_3d(sector))
    {
      uint8_t face0, face1, face2;
      get_corner_ids_3d(sector, face0, face1, face2);
      const auto face0_str = face_to_string(face0);
      const auto face1_str = face_to_string(face1);
      const auto face2_str = face_to_string(face2);
      printf("sector %d is a corner sector, face ids = %s,%s,%s\n",
             sector,
             face0_str.c_str(),
             face1_str.c_str(),
             face2_str.c_str());
    }
  }
}

} // namespace core

} // namespace kalypsso
