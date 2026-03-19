// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/vof/interface_tracking_utils.h>
#include <kalypsso/core/geometry_utils.h>

#include "gtest/gtest.h"

namespace kalypsso
{

TEST(kalypsso_get_area_below_plane, volume_fraction_2d)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 0.99, 0.99 };
  real_t                   radius = sqrt(2);

  Kokkos::Array<real_t, 2> normal{ 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 0.875, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_2d_2)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 1.01, 1.01 };
  real_t                   radius = sqrt(2);

  Kokkos::Array<real_t, 2> normal{ 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 0.125, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_2d_3)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 1.01, 0.0 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 2> normal{ 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 1.0 / 4, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_2d_4)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 1.01, 0.0 };
  real_t                   radius = 1.01;

  Kokkos::Array<real_t, 2> normal{ 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 0.5, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_2d_5)
{

  Kokkos::Array<real_t, 2> O{ 0.0, 0.0 };
  Kokkos::Array<real_t, 2> M{ 0.99, 0.99 };
  real_t                   radius = sqrt(2);

  Kokkos::Array<real_t, 2> normal{ 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.08;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 46.0 / 64, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_1)
{
  // case 1
  Kokkos::Array<real_t, 3> normal{ 1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0) };
  real_t                   alpha = -sqrt(3.0) / 2.0 + 1 * sqrt(3.0) / 4;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 0.75 * 0.75 * 0.75 / 6, 1e-14);
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_2)
{

  // case 2
  Kokkos::Array<real_t, 3> normal{ -1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0 };
  real_t                   alpha = -0.0000001;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 0.5, 1e-6);
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_2_2)
{
  // case 2
  Kokkos::Array<real_t, 3> normal{ 0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
  real_t                   alpha = -sqrt(2.0) / 2.0 + sqrt(2.0) / 4;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 0.125, 1e-14);
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_2_3)
{
  // case 2
  Kokkos::Array<real_t, 3> normal{ 0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
  real_t                   alpha = -sqrt(2.0) / 2.0 + 3 * sqrt(2.0) / 4;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 1 - 0.125, 1e-14);
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_4)
{
  // case 4
  Kokkos::Array<real_t, 3> O{ 0.0, 0.0, 0.0 };
  Kokkos::Array<real_t, 3> M{ 1.01, 0.0, 0.0 };
  real_t                   radius = 1.01;

  Kokkos::Array<real_t, 3> normal{ 0.0, 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 0.5, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_4_2)
{
  // case 4
  Kokkos::Array<real_t, 3> O{ 0.0, 0.0, 0.0 };
  Kokkos::Array<real_t, 3> M{ 1.01, 0.0, 0.0 };
  real_t                   radius = 1.0;

  Kokkos::Array<real_t, 3> normal{ 0.0, 0.0, 0.0 };
  real_t                   alpha = 0.0;

  {
    get_tangent_to_sphere(O, radius, M, normal, alpha);

    const real_t delta_x = 0.04;

    // change of frame so that cell of size delta_x about M becomes a unit cube
    alpha = (alpha - normal[IX] * M[IX] - normal[IY] * M[IY]) / delta_x;

    const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

    EXPECT_NEAR(vol_frac, 0.25, 1e-14);
  }
}

TEST(kalypsso_get_area_below_plane, volume_fraction_3d_5)
{

  // case 5
  Kokkos::Array<real_t, 3> normal{ -1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0 };
  real_t                   alpha = 0.0;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 0.5, 1e-14);
}


TEST(kalypsso_get_area_below_plane, volume_fraction_3d_last)
{
  // case last
  Kokkos::Array<real_t, 3> normal{ 1.0, 1.0, 1.0 };
  real_t                   alpha = 0.0;

  const auto vol_frac = vof::compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  EXPECT_NEAR(vol_frac, 0.5, 1e-14);
}

TEST(kalypsso_shared_vof_test, compute_plane_rhs_2d)
{
  Kokkos::Array<real_t, 2> normal = {};

  { // No normal check
    normal = { 0., 0. };
    const auto C = vof::compute_plane_rhs(0.5, normal);
    EXPECT_DOUBLE_EQ(C, 0);
  }

  { // Positive well-ordered normal
    normal = { 1., 2. };
    const auto Ca = vof::compute_plane_rhs(0.25, normal);
    const auto Cb = vof::compute_plane_rhs(0.4, normal);
    const auto Cc = vof::compute_plane_rhs(0.75, normal);
    EXPECT_DOUBLE_EQ(Ca, 1.);
    EXPECT_DOUBLE_EQ(Cb, 1.3);
    EXPECT_DOUBLE_EQ(Cc, 2.);
  }

  { // Negative unordered normal
    normal = { -2., 1. };
    const auto Ca = vof::compute_plane_rhs(0.25, normal);
    const auto Cb = vof::compute_plane_rhs(0.4, normal);
    const auto Cc = vof::compute_plane_rhs(0.75, normal);
    EXPECT_DOUBLE_EQ(Ca, -1.);
    EXPECT_DOUBLE_EQ(Cb, -0.7);
    EXPECT_DOUBLE_EQ(Cc, 0.);
  }
}

TEST(kalypsso_shared_vof_test, compute_plane_rhs_3d)
{
  Kokkos::Array<real_t, 3> normal = {};
  // Using FUZZYCOMPARE on pent ang quad test cases because of newton's approx
  // Solutions found using WolframAlpha

  { // No normal check
    normal = { 0., 0., 0. };
    const auto C = vof::compute_plane_rhs(0.5, normal);
    EXPECT_DOUBLE_EQ(C, 0);
  }

  { // Flat check
    normal = { 0., 0., 2. };
    const auto C = vof::compute_plane_rhs(0.5, normal);
    EXPECT_DOUBLE_EQ(C, 1.);
  }

  { // 2D Positive well-ordered normal
    normal = { 0., 1., 2. };
    const auto Ca = vof::compute_plane_rhs(0.25, normal);
    const auto Cb = vof::compute_plane_rhs(0.4, normal);
    const auto Cc = vof::compute_plane_rhs(0.75, normal);
    EXPECT_DOUBLE_EQ(Ca, 1.);
    EXPECT_DOUBLE_EQ(Cb, 1.3);
    EXPECT_DOUBLE_EQ(Cc, 2.);
  }

  { // 2D Negative unordered normal
    normal = { 0., -2., 1. };
    const auto Ca = vof::compute_plane_rhs(0.25, normal);
    const auto Cb = vof::compute_plane_rhs(0.4, normal);
    const auto Cc = vof::compute_plane_rhs(0.75, normal);
    EXPECT_DOUBLE_EQ(Ca, -1.);
    EXPECT_DOUBLE_EQ(Cb, -0.7);
    EXPECT_DOUBLE_EQ(Cc, 0.);
  }

  { // Positive well-ordered normal and nx + ny < nz
    normal = { 1., 2., 3.5 };
    real_t C;

    C = vof::compute_plane_rhs(0.01, normal); // tri
    EXPECT_DOUBLE_EQ(C, 0.7488872387218507);

    C = vof::compute_plane_rhs(0.1, normal); // quad A
    EXPECT_DOUBLE_EQ(C, 1.6474609652039003);

    C = vof::compute_plane_rhs(0.2, normal); // pent
    EXPECT_TRUE(FUZZYCOMPARE(C, 2.1485629016669618));

    C = vof::compute_plane_rhs(0.45, normal); // quad B
    EXPECT_DOUBLE_EQ(C, 3.075);

    C = vof::compute_plane_rhs(0.55, normal); // opp quad B
    EXPECT_DOUBLE_EQ(C, 3.425);

    C = vof::compute_plane_rhs(0.8, normal); // opp pent
    EXPECT_TRUE(FUZZYCOMPARE(C, 4.3514370983330382));

    C = vof::compute_plane_rhs(0.9, normal); // opp quad A
    EXPECT_DOUBLE_EQ(C, 4.8525390347960997);

    C = vof::compute_plane_rhs(0.99, normal); // opp tri
    EXPECT_DOUBLE_EQ(C, 5.7511127612781493);
  }

  { // Positive well-ordered normal and nx + ny > nz
    normal = { 1., 3., 3.5 };
    real_t C;

    C = vof::compute_plane_rhs(0.01, normal); // tri
    EXPECT_DOUBLE_EQ(C, 0.8572618882313395);

    C = vof::compute_plane_rhs(0.1, normal); // quad A
    EXPECT_DOUBLE_EQ(C, 1.9200938936093861);

    C = vof::compute_plane_rhs(0.35, normal); // pent
    EXPECT_TRUE(FUZZYCOMPARE(C, 3.1961420535807157));

    C = vof::compute_plane_rhs(0.45, normal); // hexa
    EXPECT_TRUE(FUZZYCOMPARE(C, 3.5706216414310085));

    C = vof::compute_plane_rhs(0.55, normal); // opp hexa
    EXPECT_TRUE(FUZZYCOMPARE(C, 3.9293783585689915));

    C = vof::compute_plane_rhs(0.65, normal); // opp pent
    EXPECT_TRUE(FUZZYCOMPARE(C, 4.3038579464192843));

    C = vof::compute_plane_rhs(0.9, normal); // opp quad A
    EXPECT_DOUBLE_EQ(C, 5.5799061063906139);

    C = vof::compute_plane_rhs(0.99, normal); // opp tri
    EXPECT_DOUBLE_EQ(C, 6.6427381117686605);
  }

  { // Negative unordered normal and nx + ny < nz
    normal = { -3.5, 1., 2. };
    real_t C;

    C = vof::compute_plane_rhs(0.01, normal); // tri
    EXPECT_DOUBLE_EQ(C, -2.7511127612781493);

    C = vof::compute_plane_rhs(0.1, normal); // quad A
    EXPECT_DOUBLE_EQ(C, -1.8525390347960997);

    C = vof::compute_plane_rhs(0.2, normal); // pent
    EXPECT_TRUE(FUZZYCOMPARE(C, -1.3514370983330382));

    C = vof::compute_plane_rhs(0.45, normal); // quad B
    EXPECT_DOUBLE_EQ(C, -0.425);
  }

  { // Negative unordered normal and nx + ny > nz
    normal = { -3.5, 1., 3. };
    real_t C;

    C = vof::compute_plane_rhs(0.01, normal); // tri
    EXPECT_DOUBLE_EQ(C, -2.6427381117686605);

    C = vof::compute_plane_rhs(0.1, normal); // quad A
    EXPECT_DOUBLE_EQ(C, -1.5799061063906139);

    C = vof::compute_plane_rhs(0.35, normal); // pent
    EXPECT_TRUE(FUZZYCOMPARE(C, -0.3038579464192843));

    C = vof::compute_plane_rhs(0.45, normal); // hexa
    EXPECT_TRUE(FUZZYCOMPARE(C, 0.07062164143100854));
  }
}

} // namespace kalypsso
