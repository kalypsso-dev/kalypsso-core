// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/vof/youngs.h>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_shared_vof_test, youngs_normal_2d)
{
  const Kokkos::Array<real_t, 2> d = { 2., 3. };
  vof::NormalNeighbors<2>        f;

  { // Only ones on east side
    f = {};
    f[2] = f[5] = f[8] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], -1.);
    EXPECT_DOUBLE_EQ(normal[1], 0);
  }

  { // Only ones on west side
    f = {};
    f[0] = f[3] = f[6] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], 1.);
    EXPECT_DOUBLE_EQ(normal[1], 0);
  }

  { // Diagonal, ones on the south-east
    f = {};
    f[1] = f[2] = f[5] = 1.;
    f[0] = f[4] = f[8] = 0.5;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], -0.8320502943378436);
    EXPECT_DOUBLE_EQ(normal[1], 0.5547001962252291);
  }

  { // Ring of ones
    f = {};
    f[0] = f[1] = f[2] = f[3] = f[5] = f[6] = f[7] = f[8] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], 0);
    EXPECT_DOUBLE_EQ(normal[1], 0);
  }
}

TEST(kalypsso_shared_vof_test, youngs_normal_3d)
{
  const Kokkos::Array<real_t, 3> d = { 2., 3., 4. };
  vof::NormalNeighbors<3>        f;

  { // Only ones on east side
    f = {};
    f[2] = f[5] = f[8] = f[11] = f[14] = f[17] = f[20] = f[23] = f[26] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], -1.);
    EXPECT_DOUBLE_EQ(normal[1], 0);
    EXPECT_DOUBLE_EQ(normal[2], 0);
  }

  { // Only ones on west side
    f = {};
    f[0] = f[3] = f[6] = f[9] = f[12] = f[15] = f[18] = f[21] = f[24] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], 1.);
    EXPECT_DOUBLE_EQ(normal[1], 0);
    EXPECT_DOUBLE_EQ(normal[2], 0);
  }

  { // Diagonal, ones on the top-south-east
    f = {};
    f[11] = f[19] = f[20] = f[23] = 1.;
    f[2] = f[10] = f[14] = f[18] = f[22] = f[26] = 0.5;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], -0.7682212795973758);
    EXPECT_DOUBLE_EQ(normal[1], 0.5121475197315838);
    EXPECT_DOUBLE_EQ(normal[2], -0.3841106397986879);
  }

  { // Ring of ones
    f = {};
    f[0] = f[1] = f[2] = f[3] = f[4] = f[5] = f[6] = f[7] = f[8] = 1.;
    f[9] = f[10] = f[11] = f[12] = f[14] = f[15] = f[16] = f[17] = 1.;
    f[18] = f[19] = f[20] = f[21] = f[22] = f[23] = f[24] = f[25] = f[26] = 1.;
    const auto normal = vof::youngs_normal(f, d);
    EXPECT_DOUBLE_EQ(normal[0], 0);
    EXPECT_DOUBLE_EQ(normal[1], 0);
    EXPECT_DOUBLE_EQ(normal[2], 0);
  }
}

TEST(kalypsso_shared_vof_test, youngs_advect_2d)
{
  Kokkos::Array<real_t, 2> normal;
  real_t                   C;
  Kokkos::Array<real_t, 2> d = { 1., 2. };

  { // normal = 0
    normal = { 0., 0. };
    C = 0.;
    real_t volume;

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.1);
    EXPECT_DOUBLE_EQ(volume, 0);

    volume = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.1);
    EXPECT_DOUBLE_EQ(volume, 0);

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.3);
    EXPECT_DOUBLE_EQ(volume, 0.3619288125423016);

    volume = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.3);
    EXPECT_DOUBLE_EQ(volume, 0.3619288125423016);

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.9);
    EXPECT_DOUBLE_EQ(volume, 0.5555555555555555);

    volume = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.9);
    EXPECT_DOUBLE_EQ(volume, 0.5555555555555555);
  }

  { // any normal, essentially a wrapper around compute_volume_fraction_of_unit_cube_below_plane, so
    // no need for extensive testing
    normal = { -1., 2. };
    real_t volumeX, volumeY;

    C = -2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 0.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.03125);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.53125);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 3.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.78125);
    EXPECT_DOUBLE_EQ(volumeY, 0.5);

    C = 4.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 1.);
    EXPECT_DOUBLE_EQ(volumeY, 1.);
  }
}

TEST(kalypsso_shared_vof_test, youngs_advect_3d)
{
  Kokkos::Array<real_t, 3> normal;
  real_t                   C;
  Kokkos::Array<real_t, 3> d = { 1., 2., 3. };

  { // normal = 0
    normal = { 0., 0., 0. };
    C = 0.;
    real_t volume;

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.1);
    EXPECT_DOUBLE_EQ(volume, 0);

    volume = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.1);
    EXPECT_DOUBLE_EQ(volume, 0);

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.3);
    EXPECT_DOUBLE_EQ(volume, 0.4133596500350424);

    volume = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.3);
    EXPECT_DOUBLE_EQ(volume, 0.4133596500350424);

    volume = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.9);
    EXPECT_DOUBLE_EQ(volume, 0.5555555555555555);

    volume = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.9);
    EXPECT_DOUBLE_EQ(volume, 0.5555555555555555);
  }

  { // any normal, essentially a wrapper around compute_volume_fraction_of_unit_cube_below_plane, so
    // no need for extensive testing
    normal = { -1., 2., 0. };
    real_t volumeX, volumeY;

    C = -2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 0.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.03125);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.53125);
    EXPECT_DOUBLE_EQ(volumeY, 0);

    C = 3.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.78125);
    EXPECT_DOUBLE_EQ(volumeY, 0.5);

    C = 4.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeY = vof::youngs_advect(0.5, C, normal, d, IY, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 1.);
    EXPECT_DOUBLE_EQ(volumeY, 1.);
  }

  { // any normal, essentially a wrapper around compute_volume_fraction_of_unit_cube_below_plane, so
    // no need for extensive testing
    normal = { -1., 2., 1. };
    real_t volumeX, volumeZ;

    C = -2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeZ = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0);
    EXPECT_DOUBLE_EQ(volumeZ, 0);

    C = 0.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeZ = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 8.6805555555555555e-4);
    EXPECT_DOUBLE_EQ(volumeZ, 0);

    C = 2.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeZ = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.1883680555555555);
    EXPECT_DOUBLE_EQ(volumeZ, 0.0234375);

    C = 3.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeZ = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.40625);
    EXPECT_DOUBLE_EQ(volumeZ, 0.21875);

    C = 4.;
    volumeX = vof::youngs_advect(0.5, C, normal, d, IX, false, 0.25);
    volumeZ = vof::youngs_advect(0.5, C, normal, d, IZ, true, 0.25);
    EXPECT_DOUBLE_EQ(volumeX, 0.6553819444444444);
    EXPECT_DOUBLE_EQ(volumeZ, 0.46875);
  }
}

} // namespace kalypsso
