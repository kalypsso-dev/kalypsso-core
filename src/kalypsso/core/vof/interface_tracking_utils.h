// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * \file interfacetracking_utils.h
 *
 * Define helper routines associated to equations (11) and (13) of the following reference
 * Volume-of-Fluid Interface Tracking with Smoothed Surface Stress Methods for Three-Dimensional
 * Flows, Gueyffier et al, JCP 152, (1999) 423-456.
 * https://doi.org/10.1006/jcph.1998.6168
 *
 * This article describes how to compute area (2d) and volume (3d) of the intersection of o
 * rectangular box (or mesh cell) with a half-space defined by cartesian inequation \f$ nx*x + ny*y
 * + nz*z <= alpha\f$. This area (resp. volume) can be recasted/normalized to be a volume fraction.
 *
 * Here we define routine for
 * - the forward evaluation : compute area/volume from the half-space inequation RHS (noted alpha in
 * Gueffier)
 * - the backward evaluation : compute half-space inequation RHS from the area/volume
 */
#ifndef KALYPSSO_CORE_INTERFACE_TRACKING_UTILS_H_
#define KALYPSSO_CORE_INTERFACE_TRACKING_UTILS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

namespace kalypsso
{

namespace vof
{

// ================================================================================================
// ================================================================================================
// ================================================================================================
//
// Forward evaluation of equation (11) and (13) of Gueyffier et al
//
// ================================================================================================
// ================================================================================================
// ================================================================================================


// =======================================================================================
// =======================================================================================
/**
 * Compute the volume fraction of a unit cube (centered at origin) intersected by a half-space of
 * equation (nx*x + ny*y + nz*z >= alpha).
 *
 * the normal vector n (nx, ny, nz) doesn't need to be a unit vector
 *
 * Given a planar interface between 2 regions (labelled "0" and "1"), and defining
 * n (nx,ny,nz) as the normal vector oriented from 0 (inside) to 1 (outside), compute the volume
 * fraction of a cube overlapped by region 0.
 *
 *        ____________
 *  0.5  |      \    |
 *       |       \ 1 |
 *       |        \  |
 *       |    0    \ |
 *       |          \|
 *       |           |
 * -0.5  |___________|
 *
 *     -0.5         0.5
 *
 * Reference:
 * Volume-of-Fluid Interface Tracking with Smoothed Surface Stress Methods for Three-Dimensional
 * Flows, Gueyffier et al, JCP 152, (1999) 423-456.
 * https://doi.org/10.1006/jcph.1998.6168
 *
 * See Fig 1 (2d case) and Fig 2. (3d case).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_volume_fraction_of_unit_cube_below_plane(Kokkos::Array<real_t, dim> normal, real_t alpha)
{
  if constexpr (dim == 2)
  {
    auto & nx = normal[IX];
    auto & ny = normal[IY];

    real_t area;

    // apply change of frame (translation)
    // so that origin is the lower left corner of the unit cube, i.e.
    // x' = x + 1/2
    // y' = y + 1/2
    alpha += (nx + ny) / KALYPSSO_NUM(2.0);

    // apply symmetry x' = 1 - x
    if (nx < KALYPSSO_NUM(0.0))
    {
      alpha -= nx;
      nx = -nx;
    }

    // apply symmetry y' = 1 - y
    if (ny < KALYPSSO_NUM(0.0))
    {
      alpha -= ny;
      ny = -ny;
    }

    // now we are left with the canonical situation where both nx and ny are positive or zero

    const auto na = min(nx, ny);
    const auto nb = max(nx, ny);

    // Formulations derived from eq 11 of Gueyffier et al. rewritten to limit floating point
    // operations imprecisions

    // alpha negative means the line is below the unit square lower left corner, so no intersection
    // the lower left half space
    if (alpha <= KALYPSSO_NUM(0.0))
    {
      area = KALYPSSO_NUM(0.0);
    }
    // alpha smaller that both of the normal coordinates, the line crosses the left and bottom
    // sides, creating a triangle
    else if (KALYPSSO_NUM(0.0) <= alpha && alpha <= na)
    {
      area = alpha * alpha / (2 * nx * ny);
    }
    // alpha between the normal coordinates, the line crosses either top-bottom (ny < nx) or
    // left-right (ny > nx) creating a trapezoid
    else if (na <= alpha && alpha <= nb)
    {
      area = (alpha - 0.5 * na) / nb;
    }
    // alpha larger than the largest normal coordinate but smaller than the sum, the line crosses
    // the top and right sides, creating a 5-sided polygon (whose complementary is a triangle)
    else if (nb <= alpha && alpha <= nx + ny)
    {
      area = KALYPSSO_NUM(1.0) - (nx + ny - alpha) * (nx + ny - alpha) / (2 * nx * ny);
    }
    // alpha larger than nx+ny means the line is above the unit square upper right corner, so the
    // unit square is fully contained in the lower left half space
    else
    {
      area = KALYPSSO_NUM(1.0);
    }

    // return volume fraction (between 0 and 1)
    return clamp(area, KALYPSSO_NUM(0.0), KALYPSSO_NUM(1.0));
  }
  else if constexpr (dim == 3)
  {
    // see equation (13) in Gueyffier et al.

    auto & nx = normal[IX];
    auto & ny = normal[IY];
    auto & nz = normal[IZ];

    // apply change of frame (translation)
    // so that origin is the lower left corner of the unit cube, i.e.
    // x' = x + 1/2
    // y' = y + 1/2
    // apply symmetries:
    // x-> 1-x
    // y-> 1-y
    // z-> 1-z
    real_t al = alpha + (nx + ny + nz) / KALYPSSO_NUM(2.0) + max(KALYPSSO_NUM(0.0), -nx) +
                max(KALYPSSO_NUM(0.0), -ny) + max(KALYPSSO_NUM(0.0), -nz);

    // no intersection at all
    if (al <= KALYPSSO_NUM(0.0))
      return KALYPSSO_NUM(0.0);

    // maximum value of nx*x+ny*y+nz*z inside the unit cube
    // noted alpha_max in Gueyffier et al.
    real_t tmp = fabs(nx) + fabs(ny) + fabs(nz);
    if (al >= tmp)
      return KALYPSSO_NUM(1.0);
    if (tmp < KALYPSSO_NUM(1e-10))
      return KALYPSSO_NUM(0.0);

    real_t n1 = fabs(nx) / tmp;
    real_t n2 = fabs(ny) / tmp;
    real_t n3 = fabs(nz) / tmp;
    al = max(KALYPSSO_NUM(0.0), min(KALYPSSO_NUM(1.0), al / tmp));

    real_t al0 = min(al, KALYPSSO_NUM(1.0) - al);
    real_t b1 = min(n1, n2);
    real_t b3 = max(n1, n2);
    real_t b2 = n3;

    // apply circular permutation to make sure to end in canonical situation where
    // b1 <= b2 <= b3
    if (b2 < b1)
    {
      tmp = b1;
      b1 = b2;
      b2 = tmp;
    }
    else if (b2 > b3)
    {
      tmp = b3;
      b3 = b2;
      b2 = tmp;
    }

    real_t b12 = b1 + b2;
    real_t bm = min(b12, b3);
#ifdef KALYPSSO_CORE_USE_DOUBLE
    real_t pr = max(KALYPSSO_NUM(6.0) * b1 * b2 * b3, KALYPSSO_NUM(1e-50));
#else
    real_t pr = max(KALYPSSO_NUM(6.0) * b1 * b2 * b3, KALYPSSO_NUM(1e-20));
#endif

    // develop equation (13) in Gueyffier et al.

    if (al0 < b1)
    {
      // case 1
      tmp = al0 * al0 * al0 / pr;
    }
    else if (al0 < b2)
    {
      // case 2
      tmp = KALYPSSO_NUM(0.5) * al0 * (al0 - b1) / (b2 * b3) + b1 * b1 * b1 / pr;
    }
    else if (al0 < bm)
    {
      // case 3

      // clang-format off
      tmp = (al0 * al0 * (KALYPSSO_NUM(3.0) * b12 - al0) +
             b1 * b1 * (b1 - KALYPSSO_NUM(3.0) * al0) +
             b2 * b2 * (b2 - KALYPSSO_NUM(3.0) * al0)) /
            pr;
      // clang-format on
    }
    else if (b12 < b3) // we also have al0 >= bm (= b1+b2)
    {
      // case 4
      tmp = (al0 - KALYPSSO_NUM(0.5) * bm) / b3;
    }
    else if (al0 >= KALYPSSO_NUM(0.5) /*and b2 >= 0.5 and b3 >= 0.5*/)
    {
      // we have b1+b2 >= b3 = 1 - (b1+b2)
      // thus al0 >= b1+b2 >= 0.5
      // and actually al0 = 0.5 (can't be strictly larger than 0.5)
      // in that case, the plane is passing through the center of the unit cube

      // case 5
      tmp = 0.5;
    }
    else
    {
      // last case
      // clang-format off
      tmp = (al0 * al0 * (KALYPSSO_NUM(3.0) - KALYPSSO_NUM(2.0) * al0) +
             b1 * b1 * (b1 - KALYPSSO_NUM(3.0) * al0) +
             b2 * b2 * (b2 - KALYPSSO_NUM(3.0) * al0) +
             b3 * b3 * (b3 - KALYPSSO_NUM(3.0) * al0)) /
            pr;
      // clang-format on
    }

    real_t volume = al <= KALYPSSO_NUM(0.5) ? tmp : KALYPSSO_NUM(1.0) - tmp;

    return clamp(volume, KALYPSSO_NUM(0.0), KALYPSSO_NUM(1.0));
  }
} // compute_volume_fraction_of_unit_cube_below_plane

// =======================================================================================
// =======================================================================================
/**
 * Compute the volume fraction of a rectangle (2D or 3D) intersected by a semi-space of
 * equation (nx*x + ny*y + nz*z >= alpha).
 *
 * the normal vector n (nx, ny, nz) doesn't need to be a unit vector
 *
 * Given a planar interface between 2 regions (labelled "0" and "1"), and defining
 * n (nx,ny,nz) as the normal vector oriented from 0 (inside) to 1 (outside), compute the volume
 * fraction of a rectangular prism overlapped by region 0.
 *
 *        ____________
 *   y1  |      \    |
 *       |       \ 1 |
 *       |        \  |
 *       |    0    \ |
 *       |          \|
 *       |           |
 *   y0  |___________|
 *
 *      x0          x1
 *
 * This transforms the volume into its unit counterpart and then calls
 * `compute_volume_fraction_of_unit_cube_below_plane`
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_volume_fraction_of_rect_below_plane(Kokkos::Array<real_t, dim> normal,
                                            real_t                     alpha,
                                            Kokkos::Array<real_t, dim> p0,
                                            Kokkos::Array<real_t, dim> p1)
{
  // We want x', y' and z' to vary between -0.5 and 0.5 such that nx'*x' + ny'*y' + nz'*z' = alpha'
  // corresponds to the same semi-space as nx*x + ny*y + nz*z >= alpha but translated and properly
  // scaled.

  // We first translate it so that each coordinate c varies between -(c1 - c0)/2 and (c1 - c0)/2
  const auto translation = (p0 + p1) * 0.5;
  for (uint8_t i = 0; i < dim; i++)
    alpha -= normal[i] * translation[i];

  // We then rescale the normal so that each coordinate varies between -0.5 and 0.5
  const auto scale = p1 - p0;
  for (uint8_t i = 0; i < dim; i++)
    normal[i] *= scale[i];

  // Then we compute
  return compute_volume_fraction_of_unit_cube_below_plane(normal, alpha);

  // Because we compute a volume fraction, there are no extra computations to be done.
} // compute_volume_fraction_of_rect_below_plane

// ================================================================================================
// ================================================================================================
// ================================================================================================
//
// Backward evaluation of equation (11) and (13) of Gueyffier et al
//
// ================================================================================================
// ================================================================================================
// ================================================================================================

// ================================================================================================
// PLANE FINDER
// ================================================================================================

#define KALYPSSO_YOUNGS_NEWTON_MAX_ITERATIONS 50
#define KALYPSSO_YOUNGS_NEWTON_PRECISION FUZZY_THRESHOLD_F

/**
 * \brief Finds the value of C (right-hand-side of plane inequation) in the pentagonal case using
 * Newton's method (solution of eq. (13))
 */
KOKKOS_FUNCTION real_t
compute_plane_rhs_pentagonal_newton(const real_t                   rhs,
                                    real_t                         guess,
                                    const Kokkos::Array<real_t, 3> normal);
/**
 * \brief Finds the value of C (right-hand-side of plane inequation) in the hexagonal case using
 * Newton's method (solution of eq. (13))
 */
KOKKOS_FUNCTION real_t
compute_plane_rhs_hexagonal_newton(const real_t                   rhs,
                                   real_t                         guess,
                                   const Kokkos::Array<real_t, 3> normal);

/**
 * \brief Computes C such that half-space nx.x + ny.y + nz.z <= C intersected with a unit cube gives
 * a volume of (volumic fraction) alpha.
 *
 * \param[in] alpha volume fraction (of the upwind material to the interface, opposite direction of
 * the normal vector)
 * \param[in] normal Interface unit normal vector
 *
 *               /
 * +------------x-----+
 * |           /      |
 * |     n _  /       |
 * |      |\ /        |
 * |        X         |
 * |       /  alpha   |
 * |      /           |
 * |     /            |
 * +----x-------------+
 *     /
 */
template <size_t dim>
KOKKOS_FUNCTION real_t
compute_plane_rhs(const real_t alpha, const Kokkos::Array<real_t, dim> normal)
{
  static constexpr int sdim = static_cast<int>(dim);

  // If the normal is null, it is impossible to compute C
  if constexpr (dim == 2)
  {
    if (ISFUZZYNULL(normal[0]) && ISFUZZYNULL(normal[1]))
      return 0.;
  }
  else if constexpr (dim == 3)
  {
    if (ISFUZZYNULL(normal[0]) && ISFUZZYNULL(normal[1]) && ISFUZZYNULL(normal[2]))
      return 0.;
  }

  // Values on which the computation will be done
  real_t                     alpha_ = alpha;
  Kokkos::Array<real_t, dim> normal_ = normal;
  Kokkos::Array<bool, dim>   syms = init_kokkos_array<bool, dim>(false);

  // Changing alpha_ and normal_ so we work in an easier state (alpha' < 1/2 and 0 < nx' < ny' <
  // nz')
  {
    // Removing half of possible cases
    if (alpha > KALYPSSO_NUM(0.5))
    {
      alpha_ = ONE_F - alpha_;
      normal_ = -ONE_F * normal_;
    }

    // Symmetries along axes
    for (int i = 0; i < sdim; i++)
      if (normal_[i] < 0)
      {
        syms[i] = true;
        normal_[i] *= -1;
      }

    // Symmetries along diagonals to get the proper axis order
    if constexpr (dim == 2)
    {
      const int i_min = normal_[0] < normal_[1] ? 0 : 1;

      normal_ = { normal_[i_min], normal_[1 - i_min] };
      syms = { syms[i_min], syms[1 - i_min] };
    }
    else if constexpr (dim == 3)
    {
      const int i_min = normal_[0] < normal_[1] ? (normal_[0] < normal_[2] ? 0 : 2)
                                                : (normal_[1] < normal_[2] ? 1 : 2);
      const int i0 = (i_min + 1) % sdim;
      const int i1 = (i_min + 2) % sdim;
      const int i_max = normal_[i0] > normal_[i1] ? i0 : i1;

      normal_ = { normal_[i_min], normal_[3 - i_min - i_max], normal_[i_max] };
      syms = { syms[i_min], syms[3 - i_min - i_max], syms[i_max] };
    }
  }

  // We now have an easier state to work with and can now properly compute C
  // This is done by checking thresholds indicating in which case we are located, as defined in:
  // Volume-of-Fluid Interface Tracking with Smoothed Surface Stress Methods for Three-Dimensional
  // Flows, Gueyffier et al, JCP 152, (1999) 423-456.
  // https://doi.org/10.1006/jcph.1998.6168
  real_t C;

  if constexpr (dim == 2) // 2D cases
  {
    const auto [nx, ny] = normal_;

    /**
     * CASES:
     *
     *   +-------+    +-------+    +-------+
     *   |       |    |       |    ¨-_     |
     *  -----------   -_      |    |  ¨-_  |
     *   |       |    | ¨-_   |    |     ¨-_
     *   +-------+    +----¨--+    +-------+
     *   horizontal   triangular   trapezoid
     */

    // Everything is in relation with eq. (11)
    const real_t rhs_area = 2 * nx * ny * alpha_; // Right hand side
    const real_t lhs_tri_lim = nx * nx; // Max value of left hand side to be in the triangular case

    // clang-format off
    if (ISFUZZYNULL(nx))                // horizontal interface case
      C = ny * alpha_;                  //
    else if (rhs_area < lhs_tri_lim)    // triangular case
      C = sqrt(rhs_area);               //
    else                                // trapezoid case
      C = nx * HALF_F + ny * alpha_;    //
                                        // no extra cases because the pentagon case is computed
                                        // only if alpha > 0.5, and the vertical interface is
                                        // impossible (the normal is not null and ny >= nx > 0)
    // clang-format on
  }
  else if constexpr (dim == 3) // 3D cases, see fig. 3 and 4
  {
    const auto [nx, ny, nz] = normal_;

    /**
     * CASES:
     *
     * triangular:    quadrilateral A:            pentagonal:
     *  pre fig. 3-a    between fig. 3-a and 3-b    between fig. 3-b and 3-c
     *  pre fig. 4-a    between fig. 4-a and 4-b    between fig. 4-b and 4-c
     *
     * hexagonal:                 quadrilateral B:
     *  between fig. 4-c and 4-d    between fig. 3-c and 3-d
     *
     * 2D cases (degenerate cases):
     *  nx = 0
     */

    // Everything is in relation with eq. (13)
    const real_t rhs_area = 6 * nx * ny * nz * alpha_; // Right hand side
    const real_t lhs_tri_lim = nx * nx * nx; // Max value of left hand side to be in triangular case
    const real_t lhs_quadA_lim = // Max value of left hand side to be in quadrilateral A case
      ny * ny * ny - (ny - nx) * (ny - nx) * (ny - nx);

    // clang-format off
    if (ISFUZZYNULL(nx))                                                                // 2D degenerate cases
    {                                                                                   //
      // Reduced to 2D cases                                                            //
      const real_t rhs_area_2d = 2 * ny * nz * alpha_;                                  //
      const real_t lhs_tri_lim_2d = ny * ny;                                            //
                                                                                        //
      if (ISFUZZYNULL(ny))                                                              // 2D horizontal case
        C = nz * alpha_;                                                                //
      else if (rhs_area_2d < lhs_tri_lim_2d)                                            // 2D triangular case
        C = sqrt(rhs_area_2d);                                                          //
      else                                                                              // 2D trapezoid case
        C = ny * HALF_F + nz * alpha_;                                                  //
    }                                                                                   //
    else if (rhs_area < lhs_tri_lim)                                                    // triangular case
      C = pow(rhs_area, ONE_THIRD_F);                                                   //
    else if (rhs_area < lhs_quadA_lim)                                                  // quadrilateral A case
    {                                                                                   //
      // This is computable by finding the area using 2 methods:                        //
      // - using the normal                                                             //
      // - using the normal with nx = 0                                                 //
      // and solving for C                                                              //
      C = (3 * nx + sqrt(72 * ny * nz * alpha_ - 3 * nx * nx)) / KALYPSSO_NUM(6.0);     //
    }                                                                                   //
    else if (nx + ny < nz)                                                              // fig. 3 extra cases
    {                                                                                   //
      const real_t C_quadB_lim = nx + ny; // Limit to be in quadB case                  //
      C = nx * HALF_F + ny * HALF_F + nz * alpha_;                                      // quadrilateral B case
                                                                                        //
      if (C < C_quadB_lim)                                                              // pentagonal case
        C = compute_plane_rhs_pentagonal_newton(rhs_area, nx + ny, normal_);            //
    }                                                                                   //
    else                                                                                // fig. 4 extra cases
    {                                                                                   //
      const real_t lhs_pent4_lim = // Limit to be in pentagonal case                    //
        nz * nz * nz - (nz - nx) * (nz - nx) * (nz - nx)                                //
                     - (nz - ny) * (nz - ny) * (nz - ny);                               //
                                                                                        //
      if (rhs_area < lhs_pent4_lim)                                                     // pentagonal case
        C = compute_plane_rhs_pentagonal_newton(rhs_area, nz, normal_);                 //
      else                                                                              // hexagonal case
        C = compute_plane_rhs_hexagonal_newton(rhs_area, (nx + ny + nz) / KALYPSSO_NUM(3.0), normal_); //
    }                                                                                   //
    // clang-format on
  }

  // We revert the changes of the final value in reverse order
  {
    // Symmetries along the diagonal do not change C

    // Symmetries along an axis change C
    for (int i = 0; i < sdim; i++)
      if (syms[i])
        C -= normal_[i];

    // If we computed the other side, we need to revert it as well
    if (alpha > KALYPSSO_NUM(0.5))
      C *= -ONE_F;
  }

  return C;

} // compute_plane_rhs

/**
 * \brief Computes C such that nx.x + ny.y + nz.z <= C intersected with a rectangular prism gives a
 * fractional volume of alpha
 *
 *               /
 * +------------x-----+ <
 * |           /      |
 * |     n _  /       |
 * |      |\ /        |
 * |        X         | d[1]
 * |       /  alpha   |
 * |      /           |
 * |     /            |
 * +----x-------------+ <
 *     /
 * ^       d[0]       ^
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_plane_rhs(const real_t                     alpha,
                  const Kokkos::Array<real_t, dim> normal,
                  const Kokkos::Array<real_t, dim> d)
{
  // We want x, y and z to vary from 0 to 1 instead of 0 to dx, dy or dz. To change that without
  // modifying C, one can simply multiply each coordinate of the normal with the cell size.
  // nx.x + ny.y + nz.z = nx'.x' + ny'.y' + nz'.z' = nx.dx.x' + ny.dy.y' + nz.dz.z'
  return compute_plane_rhs(alpha, normal * d);
}

} // namespace vof

} // namespace kalypsso

#endif // KALYPSSO_CORE_INTERFACE_TRACKING_UTILS_H_
