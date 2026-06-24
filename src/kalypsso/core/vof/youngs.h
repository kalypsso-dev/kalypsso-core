// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file youngs.h
 *
 * \brief Collection of methods to help in computing youngs interface reconstruction
 *
 * Reference:
 * - An interface tracking method for a 3D Eulerian hydrodynamics code, D. L. Youngs, AWRE technical
 * report /44/92/35.
 *   https://www.researchgate.net/publication/245345562_An_interface_tracking_method_for_a_3D_Eulerian_hydrodynamics_code
 */

#ifndef KALYPSSO_CORE_VOF_YOUNGS_H_
#define KALYPSSO_CORE_VOF_YOUNGS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/utils_block.h>

#include <kalypsso/core/vof/interface_tracking_utils.h>

namespace kalypsso
{

namespace vof
{

// ================================================================================================
// ================================================================================================
// NORMAL
// ================================================================================================
// ================================================================================================

/**
 * \brief Indicates the list of neighbors in a 3x3 or 3x3x3 stencil used in normal computation
 */
template <size_t dim>
using NormalNeighbors = Kokkos::Array<real_t, dim == 2 ? 9 : 27>;

/**
 * \brief Computes the interface normal vector from a size-3 stencil (9 cells in 2D or 27 in 3D)
 * using (Parker-)Youngs method.
 *
 * \param[in] f The volume fraction values of current cell and its neighbors listed in a X-Y-Z order
 * \param[in] dxyz Vector of cell sizes
 *
 * +------+------+------+
 * |      |      |      |
 * | f[6] | f[7] | f[8] |
 * |      |      |      |
 * +------+------+------+
 * |      |      |      |
 * | f[3] | f[4] | f[5] |
 * |      |      |      |
 * +------+------+------+ <     y
 * |      |      |      |       ^
 * | f[0] | f[1] | f[2] | d[1]  |
 * |      |      |      |       |
 * +------+------+------+ <     +----> x
 *               ^ d[0] ^
 *
 * References:
 *
 * - An interface tracking method for a 3D Eulerian hydrodynamics code, D. L. Youngs, AWRE technical
 *   report /44/92/35.
 *
https://www.researchgate.net/publication/245345562_An_interface_tracking_method_for_a_3D_Eulerian_hydrodynamics_code
 *   here is explained how parameter alpha (2D) and beta/gamma (3D) should be chosen.
 *
 * - Second order accurate volume-of-fluid algorithms for tracking material interfaces, J.E. Pilliod
 *   and E. G. Puckett, JCP, vol 199, 2 (2004), pp. 465-502.
 *   https://doi.org/10.1016/j.jcp.2003.12.023.
 *   See section 2.4
 *
 * - Volume of fluid interface reconstruction methods for multi-material problems, D. J. Benson,
 *   Appl. Mech. Rev. Mar 2002, 55(2): 151-165 (15 pages)
 *   https://doi.org/10.1115/1.1448524
 *   See section 5.2
 *
 * - Interface reconstruction with least-square fit and split Eulerian–Lagrangian advection, R.
 *   Scardovelli and S. Zaleski, INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN FLUIDS Int. J.
 *   Numer.  Meth. Fluids 2003; 41:251–274.  https://doi.org/10.1002/fld.431
 *   See section 2.1
 */
template <size_t dim>
KOKKOS_FUNCTION Kokkos::Array<real_t, dim>
                youngs_normal(const NormalNeighbors<dim> & f, const Kokkos::Array<real_t, dim> dxyz)
{
  Kokkos::Array<real_t, dim> normal;

  if constexpr (dim == 2)
  {
    static constexpr real_t alpha = KALYPSSO_NUM(2.0);
    const auto [dx, dy] = dxyz;
    const real_t fact = ONE_F / (KALYPSSO_NUM(2.0) + alpha);

    const real_t f_east = fact * (f[2] + alpha * f[5] + f[8]);
    const real_t f_west = fact * (f[0] + alpha * f[3] + f[6]);

    const real_t f_north = fact * (f[6] + alpha * f[7] + f[8]);
    const real_t f_south = fact * (f[0] + alpha * f[1] + f[2]);

    normal = { -(f_east - f_west) / (2 * dx), -(f_north - f_south) / (2 * dy) };
  }
  else if constexpr (dim == 3)
  {
    static constexpr real_t beta = KALYPSSO_NUM(2.0);
    static constexpr real_t gamma = KALYPSSO_NUM(4.0);
    const auto [dx, dy, dz] = dxyz;
    const real_t fact = ONE_F / (FOUR_F + FOUR_F * beta + gamma);

    // clang-format off
    const real_t f_east = fact *
      (f[2] + f[8] + f[20] + f[26] + beta * (f[5] + f[11] + f[17] + f[23]) + gamma * f[14]);
    const real_t f_west = fact *
      (f[0] + f[6] + f[18] + f[24] + beta * (f[3] + f[9] + f[15] + f[21]) + gamma * f[12]);

    const real_t f_north = fact *
      (f[6] + f[8] + f[24] + f[26] + beta * (f[7] + f[15] + f[17] + f[25]) + gamma * f[16]);
    const real_t f_south = fact *
      (f[0] + f[2] + f[18] + f[20] + beta * (f[1] + f[9] + f[11] + f[19]) + gamma * f[10]);

    const real_t f_top = fact *
      (f[18] + f[20] + f[24] + f[26] + beta * (f[19] + f[21] + f[23] + f[25]) + gamma * f[22]);
    const real_t f_bottom = fact *
      (f[0] + f[2] + f[6] + f[8] + beta * (f[1] + f[3] + f[5] + f[7]) + gamma * f[4]);
    // clang-format on

    normal = { -(f_east - f_west) / (2 * dx),
               -(f_north - f_south) / (2 * dy),
               -(f_top - f_bottom) / (2 * dz) };
  }

  auto norm = normal[IX] * normal[IX] + normal[IY] * normal[IY];
  if constexpr (dim == 3)
    norm += normal[IZ] * normal[IZ];
  norm = sqrt(norm);

  if (norm < FUZZY_THRESHOLD_F)
    return init_kokkos_array<real_t, dim>(0);
  else
    normal = normal / norm;

  for (uint8_t i = 0; i < dim; i++)
  {
    if (abs(normal[i]) < FUZZY_THRESHOLD_F)
      normal[i] = 0.;
    else if (abs(normal[i]) > KALYPSSO_NUM(1.0) - FUZZY_THRESHOLD_F)
    {
      const real_t dir = normal[i] >= 0 ? 1. : -1.;
      normal = init_kokkos_array<real_t, dim>(0);
      normal[i] = dir;
    }
  }

  return normal;

} // ypoungs_normal

/**
 * \brief Computes the Youngs normal from inside a DataArrayBlock-like object
 *
 * \param[in] f A DataArrayBlock-like object
 * \param[in] ijk A dim-array of cell coordinate inside the block it belongs to
 * \param[in] i_var The variable index (usually identifying the material volume fraction)
 * \param[in] i_oct The octant id
 * \param[in] dxyz Vector of cell sizes
 *
 */
template <typename DAB, size_t dim>
KOKKOS_FUNCTION Kokkos::Array<real_t, dim>
                youngs_normal(const DAB &                      f,
                              const coord_t<dim>               ijk,
                              const int                        i_var,
                              const int                        i_oct,
                              const Kokkos::Array<real_t, dim> dxyz)
{
  NormalNeighbors<dim> n;

  if constexpr (dim == 2)
  {
    static constexpr coord_t<dim> ix = { 1, 0 };
    static constexpr coord_t<dim> iy = { 0, 1 };

    for (int v = 0; v < static_cast<int>(n.size()); v++)
    {
      const auto ijk2 = ijk + (v % 3 - 1) * ix + (v / 3 - 1) * iy;
      n[v] = f(ijk2, i_var, i_oct);
    }
  }
  else if constexpr (dim == 3)
  {
    static constexpr coord_t<dim> ix = { 1, 0, 0 };
    static constexpr coord_t<dim> iy = { 0, 1, 0 };
    static constexpr coord_t<dim> iz = { 0, 0, 1 };

    for (int v = 0; v < static_cast<int>(n.size()); v++)
    {
      const auto ijk2 = ijk + (v % 3 - 1) * ix + ((v / 3) % 3 - 1) * iy + (v / 9 - 1) * iz;
      n[v] = f(ijk2, i_var, i_oct);
    }
  }

  return youngs_normal(n, dxyz);

} // youngs_normal

/**
 * \brief Computes the Youngs normal from inside a DataArrayBlock-like object (in-place
 * implementation).
 *
 * \param[in] f A DataArrayBlock-like object
 * \param[in] ijk A dim-array of cell coordinate inside the block it belongs to
 * \param[in] i_var The variable index (usually identifying the material volume fraction)
 * \param[in] i_oct The octant id
 * \param[in] dxyz Vector of cell sizes
 *
 * \note Same function as youngs_normal but don't use extra array for collecting neighbor values
 */
template <typename DAB, size_t dim>
KOKKOS_FUNCTION Kokkos::Array<real_t, dim>
                youngs_normal_inplace(const DAB &                      f,
                                      const coord_t<dim>               ijk,
                                      const int                        i_var,
                                      const int                        i_oct,
                                      const Kokkos::Array<real_t, dim> dxyz)
{
  Kokkos::Array<real_t, dim> normal;

  if constexpr (dim == 2)
  {
    static constexpr real_t alpha = KALYPSSO_NUM(2.0);
    const auto [dx, dy] = dxyz;
    const auto [i, j] = ijk;
    const real_t fact = ONE_F / (KALYPSSO_NUM(2.0) + alpha);

#define F(a, b) f(i + a, j + b, i_var, i_oct)

    // clang-format off
    const real_t f_east  = fact * (F( 1, -1) + alpha * F( 1, 0) + F( 1, 1));
    const real_t f_west  = fact * (F(-1, -1) + alpha * F(-1, 0) + F(-1, 1));

    const real_t f_north = fact * (F(-1,  1) + alpha * F(0,  1) + F(1,  1));
    const real_t f_south = fact * (F(-1, -1) + alpha * F(0, -1) + F(1, -1));
    // clang-format on

#undef F

    normal = { -(f_west - f_east) / (2 * dx), -(f_north - f_south) / (2 * dy) };
  }
  else if constexpr (dim == 3)
  {
    static constexpr real_t beta = KALYPSSO_NUM(2.0);
    static constexpr real_t gamma = KALYPSSO_NUM(4.0);
    const auto [dx, dy, dz] = dxyz;
    const auto [i, j, k] = ijk;
    const real_t fact = ONE_F / (FOUR_F + FOUR_F * beta + gamma);

#define F(a, b, c) f(i + a, j + b, k + c, i_var, i_oct)
    // clang-format off
    const real_t f_east =
      fact * (        F(+1, -1, -1) + F(+1, +1, -1) + F(+1, -1, +1) + F(+1, +1, +1) +
              beta * (F(+1, +0, -1) + F(+1, -1, +0) + F(+1, +1, +0) + F(+1, +0, +1)) +
              gamma * F(+1, +0, +0));
    const real_t f_west =
      fact * (        F(-1, -1, -1) + F(-1, +1, -1) + F(-1, -1, +1) + F(-1, +1, +1) +
              beta * (F(-1, +0, -1) + F(-1, -1, +0) + F(-1, +1, +0) + F(-1, +0, +1)) +
              gamma * F(-1, +0, +0));

    const real_t f_north =
      fact * (        F(-1, +1, -1) + F(+1, +1, -1) + F(-1, +1, +1) + F(+1, +1, +1) +
              beta * (F(+0, +1, -1) + F(-1, +1, +0) + F(+1, +1, +0) + F(+0, +1, +1)) +
              gamma * F(+0, +1, +0));
    const real_t f_south =
      fact * (        F(-1, -1, -1) + F(+1, -1, -1) + F(-1, -1, +1) + F(+1, -1, +1) +
              beta * (F(+0, -1, -1) + F(-1, -1, +0) + F(+1, -1, +0) + F(+0, -1, +1)) +
              gamma * F(+0, -1,  0));

    const real_t f_top =
      fact * (        F(-1, -1, +1) + F(+1, -1, +1) + F(-1, +1, +1) + F(+1, +1, +1) +
              beta * (F(+0, -1, +1) + F(-1, +0, +1) + F(+1, +0, +1) + F(+0, +1, +1)) +
              gamma * F(+0, +0, +1));
    const real_t f_bottom =
      fact * (        F(-1, -1, -1) + F(+1, -1, -1) + F(-1, +1, -1) + F(+1, +1, -1) +
              beta * (F(+0, -1, -1) + F(-1, +0, -1) + F(+1, +0, -1) + F(+0, +1, -1)) +
              gamma * F(+0, +0, -1));
    // clang-format on
#undef F

    // clang-format off
    normal = { -(f_east  - f_west)   / (2 * dx),
               -(f_north - f_south)  / (2 * dy),
               -(f_top   - f_bottom) / (2 * dz) };
    // clang-format on
  }

  auto norm = normal[IX] * normal[IX] + normal[IY] * normal[IY];
  if constexpr (dim == 3)
    norm += normal[IZ] * normal[IZ];
  norm = sqrt(norm);

  if (norm < FUZZY_THRESHOLD_F)
    return init_kokkos_array<real_t, dim>(0);
  else
    normal = normal / norm;

  for (uint8_t i = 0; i < dim; i++)
  {
    if (abs(normal[i]) < FUZZY_THRESHOLD_F)
      normal[i] = 0.;
    else if (abs(normal[i]) > KALYPSSO_NUM(1.0) - FUZZY_THRESHOLD_F)
    {
      const real_t dir = normal[i] >= 0 ? 1. : -1.;
      normal = init_kokkos_array<real_t, dim>(0);
      normal[i] = dir;
    }
  }

  return normal;

} // youngs_normal_inplace

// ================================================================================================
// ================================================================================================
// VOLUME ADVECTION
// ================================================================================================
// ================================================================================================

/**
 * \brief Returns the volume fraction of the advected volume that gets actually advected though the
 * interface
 *
 *                    /
 * +---------------+-x-----+ <      +---------------+-------+ < <
 * |               |/      |        |               |       |   gamma * d[1]
 * |          n _  /       |        |   +---------------+   |   <
 * |           |\ /|       |        |   |           |   |   |
 * |             X         | d[1]   |   | alpha         |   | d[1]
 * |            /  |       |        |   |           |   |   |
 * |           /           |        |   +---------------+   |
 * |          /    |       |        |               |       |
 * +---------x-----+-------+ <      +---------------+-------+ <
 *          /                       ^          d[0]         ^
 * ^          d[0]         ^        ^   ^ gamma * d[0]
 */
template <size_t dim>
KOKKOS_FUNCTION real_t
youngs_advect(const real_t                     alpha,
              const real_t                     C,
              const Kokkos::Array<real_t, dim> normal,
              const Kokkos::Array<real_t, dim> d,
              const ComponentIndex3D           dir,
              const bool                       right_side,
              const real_t                     full_advected_fraction)
{
  if (ISFUZZYNULL(full_advected_fraction) || ISFUZZYNULL(alpha)) // No volume is actually advected
    return 0.;

  const bool is_null = [&normal]() {
    if constexpr (dim == 2)
      return ISFUZZYNULL(normal[0]) && ISFUZZYNULL(normal[1]);
    else if constexpr (dim == 3)
      return ISFUZZYNULL(normal[0]) && ISFUZZYNULL(normal[1]) && ISFUZZYNULL(normal[2]);
  }();

  if (is_null)
  {
    // Case normal is null. If the volume is scaled down as a unit cube, then the material's volume
    // would be a smaller unit cube 'gamma' away from the cell's border.
    const real_t gamma = HALF_F * (ONE_F - pow(alpha, ONE_F / dim));
    const real_t frac = ONE_F / full_advected_fraction;

    if (full_advected_fraction < gamma) // No intersection
      return 0.;
    else if (1 - full_advected_fraction < gamma) // Entirely inside
      return alpha * frac;
    else // Intersection
    {
      if constexpr (dim == 2)
        return (1 - 2 * gamma) * (full_advected_fraction - gamma) * frac;
      else if constexpr (dim == 3)
        return (1 - 2 * gamma) * (1 - 2 * gamma) * (full_advected_fraction - gamma) * frac;
    }
  }

  Kokkos::Array<real_t, dim> p_bottom_left = {};
  Kokkos::Array<real_t, dim> p_top_right = d;
  p_bottom_left[dir] = d[dir] * (right_side ? ONE_F - full_advected_fraction : ZERO_F);
  p_top_right[dir] = d[dir] * (right_side ? ONE_F : full_advected_fraction);

  return compute_volume_fraction_of_rect_below_plane(normal, C, p_bottom_left, p_top_right);

} // youngs_advect

} // namespace vof

} // namespace kalypsso

#endif // KALYPSSO_CORE_VOF_YOUNGS_H_
