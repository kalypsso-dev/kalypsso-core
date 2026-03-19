// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RichtmyerMeshkovParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_RICHTMYER_MESHKOV_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_RICHTMYER_MESHKOV_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Richtmyer-Meshkov instability instability test parameters.
 *
 * reference : https://doi.org/10.1016/j.compfluid.2021.105158
 *
 * 3 regions:
 *
 *                    location of material interface
 *                    with sine wave shape
 *                      ||
 *                      \/
 *
 * +--------------------+----------------+--------------+
 * |                    \                |              |
 * |                     \               |              |
 * |                     |               |              |
 * |  region 2        __/    region 1    |  region 0    |
 * |  (material 1)   /     (material 0)  | (material 0) |
 * |                |                    |              |
 * |                \                    |              |
 * |                 \                   |              |
 * +-----------------+-------------------+--------------+
 */
struct RichtmyerMeshkovParams
{

  // shock-bubble problem parameters

  //! initial shock location
  real_t x_shock;

  //! average material interface location
  real_t x_interface;

  //! center box (along the y axis)
  real_t y_center;

  //! material interface parameter
  //! epsilon is the amplitude of the spatial sine perturbation
  real_t epsilon;

  RichtmyerMeshkovParams(ConfigMap const & config_map)
  {
    x_shock = config_map.getReal("richtmyer_meshkov", "x_shock", KALYPSSO_NUM(3.2));
    x_interface = config_map.getReal("richtmyer_meshkov", "x_interface", KALYPSSO_NUM(2.9));

    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    y_center = config_map.getReal("richtmyer_meshkov", "y_center", (ymin + ymax) / 2);

    epsilon = config_map.getReal("richtmyer_meshkov", "epsilon", KALYPSSO_NUM(0.2));
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  x_material(real_t y) const
  {
    return x_interface - epsilon * sin(2 * PI_F * (y + y_center));
  }

  /**
   * Compute tangent line 2d (or plane in 3d).
   *
   * In 2d the equation of the tangent line is: \f$ n_x x + n_y y = \alpha\f$
   * In 3d the equation of the tangent plane is: \f$ n_x x + n_y y + n_z z = \alpha\f$
   *
   * In pout we return the normal vector to tangent (line or plane) oriented from region 1 to
   * region 2.
   *
   * \param[in] xyz a given point (usually a cell center)
   * \param[out] normal tangent normal unit vector (nx, ny, nz)
   * \param[out] alpha
   */
  template <size_t dim>
  KOKKOS_INLINE_FUNCTION void
  get_interface_tangent(Kokkos::Array<real_t, dim> const & xyz,
                        Kokkos::Array<real_t, dim> &       normal,
                        real_t &                           alpha) const
  {
    // compute interface derivative along y
    real_t dxdy = -epsilon * 2 * PI_F * cos(2 * PI_F * (xyz[IY] + y_center));

    normal[IX] = -dxdy;
    normal[IY] = KALYPSSO_NUM(1.0);
    if constexpr (dim == 3)
      normal[IZ] = KALYPSSO_NUM(0.0);

    auto x_interf = x_material(xyz[IY]);

    alpha = normal[IX] * x_interf + normal[IY] * xyz[IY];
  }

}; // struct RichtmyerMeshkovParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_RICHTMYER_MESHKOV_PARAMS_H_
