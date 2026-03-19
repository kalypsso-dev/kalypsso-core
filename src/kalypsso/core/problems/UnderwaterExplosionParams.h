// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UnderwaterExplosionParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_UNDERWATER_EXPLOSION_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_UNDERWATER_EXPLOSION_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Underwater explosion test parameters.
 *
 * reference :
 *
 * An interface capturing scheme for modeling atomization in compressible flows, Garrick et al.,
 * Journal of Computational Physics Volume 344, 1 September 2017, Pages 260-280.
 * https://doi.org/10.1016/j.jcp.2017.04.079
 */
struct UnderwaterExplosionParams
{

  // shock-bubble problem parameters

  //! interface location
  real_t interface_loc;

  //! bubble center location
  real_t bubble_x;
  real_t bubble_y;
  real_t bubble_z;

  //! bubble radius
  real_t bubble_radius;

  UnderwaterExplosionParams(ConfigMap const & config_map)
  {
    interface_loc =
      config_map.getReal("underwater_explosion", "interface_location ", KALYPSSO_NUM(0.0));

    bubble_x = config_map.getReal("underwater_explosion", "bubble_x", KALYPSSO_NUM(0.0));
    bubble_y = config_map.getReal("underwater_explosion", "bubble_y", KALYPSSO_NUM(-0.3));
    bubble_z = config_map.getReal("underwater_explosion", "bubble_z", KALYPSSO_NUM(0.0));

    bubble_radius = config_map.getReal("underwater_explosion", "bubble_radius", KALYPSSO_NUM(0.12));
  }

}; // struct UnderwaterExplosionParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_UNDERWATER_EXPLOSION_PARAMS_H_
