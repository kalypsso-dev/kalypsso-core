// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file volume_fraction_advection_source_term_type.h
 */
#ifndef KALYPSSO_CORE_VOLUME_FRACTION_ADVECTION_SOURCE_TERM_TYPE_H_
#define KALYPSSO_CORE_VOLUME_FRACTION_ADVECTION_SOURCE_TERM_TYPE_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

namespace core
{

// clang-format off
/**
 * An enum type to represent all the different types of models
 * for volume fraction advection equation's source term.
 *
 * Valid values are:
 *
 * - ALLAIRE_KOKH : \f$ \frac{\partial \alpha}{\partial t} + u\dot\nabla \alpha = -\alpha \div u \f$
 * - KAPILA : \f$ \frac{\partial \alpha}{\partial t} + u\dot\nabla \alpha = -\alpha (1-\kappa/\kappa_0) \div u \f$ where \f$ \kappa \f$ is the mixture bulk modulus and \f$ \kappa_0 \f$ is the material bulk modulus which volume fraction is \f$ \alpha \f$
 */
BETTER_ENUM(VolFracAdvectionSourceTermType, uint8_t,
            INVALID = 0,
            ALLAIRE_KOKH = 1,
            KAPILA = 2)
// clang-format on

/**
 * Read configuration map to initialize the volume fraction advection source term type.
 */
inline VolFracAdvectionSourceTermType
get_volume_fraction_advection_source_term_type(ConfigMap const & config_map)
{
  const auto volume_fraction_advection_source_term_type_name =
    config_map.getString("volume_fraction_advection", "source_term_type", "ALLAIRE_KOKH");
  auto maybe_value = VolFracAdvectionSourceTermType::_from_string_nothrow(
    volume_fraction_advection_source_term_type_name.c_str());
  if (maybe_value)
  {
    return *maybe_value;
  }
  else
  {
    Kokkos::abort("Wrong parameter for volume_fraction_advection/source_term_type.");
  }
  return VolFracAdvectionSourceTermType::INVALID;
}

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_VOLUME_FRACTION_ADVECTION_SOURCE_TERM_TYPE_H_
