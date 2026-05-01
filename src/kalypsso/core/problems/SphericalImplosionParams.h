// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SphericalImplosionParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_SPHERICAL_IMPLOSION_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_SPHERICAL_IMPLOSION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Spherical implosion with perturbed interface test parameters.
 *
 * references :
 *
 * Sharp interface schemes for multi-material computational fluid dynamics, PhD thesis
 * Murray Cutforth, University of Cambridge, https://doi.org/10.17863/CAM.44907
 *
 * See test called "tin implosion" presented in chapter 5, section 5.3.8
 *
 * Perturbed interface is only available in 2d. In 3d we only consider two nested spheres.
 * In 2d, this class is actually quite similar to RichtmyerMeshkovParams, except the
 * cylindrical/spherical geometry.
 *
 */
struct SphericalImplosionParams
{

  //! average radius (interface is perturbed with a sine wave)
  real_t r_in;

  //! outer radius
  real_t r_out;

  //! bubble center, x coordinate.
  real_t x;

  //! bubble center, y coordinate.
  real_t y;

  //! bubble center, z coordinate.
  real_t z;

  //! sine wave amplitude of the inner interface
  real_t amplitude;

  //! sine wave frequency of the inner interface
  real_t freq;

  //! radial velocity of fluid between r_in and r_out
  //! it can positive for centrifugal (divergent) flow
  //! or     negative for centripetal (convergent) flow
  real_t v_theta;

  SphericalImplosionParams(ConfigMap const & config_map)
  {
    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(-25.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(-25.0));
    const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(-25.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(50.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    r_in = config_map.getReal("spherical_implosion", "r_in", KALYPSSO_NUM(20.0));
    r_out = config_map.getReal("spherical_implosion", "r_out", KALYPSSO_NUM(24.0));

    x = config_map.getReal("spherical_implosion", "x", (xmin + xmax) / 2);
    y = config_map.getReal("spherical_implosion", "y", (ymin + ymax) / 2);
    z = config_map.getReal("spherical_implosion", "z", (zmin + zmax) / 2);

    amplitude = config_map.getReal("spherical_implosion", "amplitude", KALYPSSO_NUM(0.4));
    freq = config_map.getReal("spherical_implosion", "freq", KALYPSSO_NUM(22.0));
    v_theta = config_map.getReal("spherical_implosion", "v_theta", KALYPSSO_NUM(0.0));
  }

  //! shell center as a Kokkos::Array
  template <size_t dim>
  KOKKOS_INLINE_FUNCTION Kokkos::Array<real_t, dim>
                         shell_center() const
  {
    Kokkos::Array<real_t, dim> xyz;
    xyz[IX] = this->x;
    xyz[IY] = this->y;
    if constexpr (dim == 3)
    {
      xyz[IZ] = this->z;
    }
    return xyz;
  }

  //! location (radius) of the inner interface
  KOKKOS_INLINE_FUNCTION
  real_t
  f(real_t theta) const
  {
    return r_in + amplitude * cos(freq * theta);
  }

  //! derivative of f versus theta
  KOKKOS_INLINE_FUNCTION
  real_t
  f_prime(real_t theta) const
  {
    return -amplitude * freq * sin(freq * theta);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  inner_interface_radius(real_t theta) const
  {
    return f(theta);
  }

  template <size_t dim>
  KOKKOS_INLINE_FUNCTION real_t
  inner_interface_radius(Kokkos::Array<real_t, dim> const & xyz) const
  {
    const auto theta = atan2(xyz[IY] - y, xyz[IX] - x);
    return f(theta);
  }

  /**
   * Compute tangent line 2d (no 3d here, interface is not perturbed ).
   *
   * In 2d the equation of the tangent line is: \f$ n_x x + n_y y = \alpha\f$
   * In 3d the equation of the tangent plane is: \f$ n_x x + n_y y + n_z z = \alpha\f$
   *
   * In output we return the normal vector to tangent (line or plane) oriented from region 1 to
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
    // let (x,y) bet on interface
    // x = r cos(theta)
    // y = r sin(theta)
    //
    // normal vector is orthogonal with components
    // n(IX) =  dydtheta =  f'(theta) sin(theta) + f(theta) cos(theta)
    // n(IY) = -dxdtheta = -f'(theta) cos(theta) + f(theta) sin(theta)

    const auto theta = atan2(xyz[IY] - y, xyz[IX] - x);

    // clang-format off
    normal[IX] =  f_prime(theta) * sin(theta) + f(theta) * cos(theta);
    normal[IY] = -f_prime(theta) * cos(theta) + f(theta) * sin(theta);
    // clang-format on

    alpha = f(theta) * (normal[IX] * cos(theta) + normal[IY] * sin(theta));
  }

}; // struct SphericalImplosionParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_SPHERICAL_IMPLOSION_PARAMS_H_
