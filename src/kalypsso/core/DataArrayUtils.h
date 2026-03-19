// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayUtils.h
 */
#ifndef KALYPSSO_CORE_DATAARRAYUTILS_H_
#define KALYPSSO_CORE_DATAARRAYUTILS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

namespace kalypsso
{

// forward declaration necessary for defining DataArrayBase
class ConfigMap;

//! enum used as a template parameter to DataArray and DataArrayBlock
enum class StorageType
{
  OWNED = 0,
  UNMANAGED = 1
};

// ============================================================================================
// ============================================================================================
// ============================================================================================
/**
 * \class DataArrayUtils
 *
 * This is to parametrize all DataArray-like class.
 * It is meant to store common parameters; currently only parameter: the capacity growth rate
 *
 */
struct DataArrayUtils
{
  static constexpr real_t m_capacity_growth_rate_default = KALYPSSO_NUM(1.0);
  static real_t           m_capacity_growth_rate;

  static void
  set_growth_rate(ConfigMap const & config_map);

  static void
  set_growth_rate(real_t growth_rate);

  //! compute allocated capacity given initial size.
  static size_t
  allocated_capacity(size_t wanted_size)
  {
    return static_cast<size_t>(
      std::floor(static_cast<real_t>(wanted_size) * m_capacity_growth_rate));
  }
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_DATAARRAYUTILS_H_
