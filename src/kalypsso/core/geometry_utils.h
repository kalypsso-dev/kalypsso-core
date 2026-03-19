// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * \file geometry_utils.h
 */
#ifndef KALYPSSO_CORE_GEOMETRY_UTILS_H_
#define KALYPSSO_CORE_GEOMETRY_UTILS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

namespace kalypsso
{

// =======================================================================================
// =======================================================================================
/**
 * Return the coefficients of the equation of the tangent to a circle (2d) / sphere (3d).
 *
 * Given a circle or sphere (center C and radius), given a point M, computes the equation of the
 * tangent line (2d) or tangent plane (3d) to that circle/sphere which normal vector is CM and such
 * that the tangent point T verifies CT = alpha CM with alpha > 0.
 *
 * In 2d the equation of the tangent line is: \f$ n_x x + n_y y = \alpha\f$
 *
 * In 3d the equation of the tangent plane is: \f$ n_x x + n_y y + n_z z = \alpha\f$
 *
 *
 * \param[in] O circle (or sphere) center coordinates vector
 * \param[in] radius circle/sphere radius
 * \param[in] M coordinates vector of point M
 * \param[out] normal normal unit vector (nx, ny, nz)
 * \param[out] alpha
 *
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION void
get_tangent_to_sphere(Kokkos::Array<real_t, dim> const & O,
                      real_t                             radius,
                      Kokkos::Array<real_t, dim> const & M,
                      Kokkos::Array<real_t, dim> &       normal,
                      real_t &                           alpha)
{
  normal = M - O;
  auto d = normal[IX] * normal[IX] + normal[IY] * normal[IY];
  if constexpr (dim == 3)
  {
    d += normal[IZ] * normal[IZ];
  }
  d = sqrt(d);

  normal[IX] /= d;
  normal[IY] /= d;
  if constexpr (dim == 3)
  {
    normal[IZ] /= d;
  }

  Kokkos::Array<real_t, dim> T;
  T[IX] = O[IX] + radius * (normal[IX]);
  T[IY] = O[IY] + radius * (normal[IY]);
  if constexpr (dim == 3)
  {
    T[IZ] = O[IZ] + radius * (normal[IZ]);
  }

  alpha = normal[IX] * T[IX] + normal[IY] * T[IY];
  if constexpr (dim == 3)
  {
    alpha += normal[IZ] * T[IZ];
  }

} // get_tangent_to_sphere

} // namespace kalypsso

#endif // KALYPSSO_CORE_GEOMETRY_UTILS_H_
