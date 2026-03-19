// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeInterfaceNormal.h
 *
 * Compute schlieren of a scalar field.
 *
 */
#ifndef KALYPSSO_CORE_INTERFACENORMALVECTORALGORITHMPARAMS_H_
#define KALYPSSO_CORE_INTERFACENORMALVECTORALGORITHMPARAMS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
/**
 * An enum type to represent all possible algorithm variant for computing interface normal vector.
 */
BETTER_ENUM(InterfaceNormalVectorAlgorithmType, uint8_t,
            INVALID = 0,
            PARKER_AND_YOUNGS = 1,
            ELVIRA = 2,
            SMOOTH_INTERFACE_FUNCTION = 3)
// clang-format on

//! Get interface normal vector computation algorithm type for input parameter file.
inline InterfaceNormalVectorAlgorithmType
get_interface_normal_vector_algorithm_type(ConfigMap const & config_map)
{
  const auto algo_name =
    config_map.getString("material", "interface_normal_vector_algorithm", "INVALID");
  auto maybe_value = InterfaceNormalVectorAlgorithmType::_from_string_nothrow(algo_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return InterfaceNormalVectorAlgorithmType::INVALID;
}

/**
 * Parker and Youngs algorithm to compute interface normal.
 *
 * references:
 * - An interface tracking method for a 3D Eulerian hydrodynamics code, D. Youngs (1987);
 * https://www.researchgate.net/publication/245345562_An_interface_tracking_method_for_a_3D_Eulerian_hydrodynamics_code
 * - Volume of fluid interface reconstruction methods for multi-material problems, D.J. Benson,
 * Appl. Mech. Rev. Mar 2002, 55(2): 151-165 (2002). https://doi.org/10.1115/1.1448524
 *
 * Interface normal is computed by \f$ n = \frac{\nabla f}{|\nabla f|} where f is the volume
 * fraction of one material
 */
struct ParkerAndYoungsParams
{
  real_t alpha;
  real_t beta;
  real_t gamma;

  ParkerAndYoungsParams(ConfigMap const & config_map)
  {
    alpha = config_map.getFloat("parker_and_youngs", "alpha", 2.0);
    beta = config_map.getFloat("parker_and_youngs", "beta", 2.0);
    gamma = config_map.getFloat("parker_and_youngs", "gamma", 4.0);
  }

}; // struct ParkerAndYoungsParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_INTERFACENORMALVECTORALGORITHMPARAMS_H_
