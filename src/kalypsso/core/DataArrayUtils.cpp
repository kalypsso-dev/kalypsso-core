// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataArrayUtils.h
 */

#include <kalypsso/core/DataArrayUtils.h>

#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

// default value is 1.0 (no growth rate)
real_t DataArrayUtils::m_capacity_growth_rate = m_capacity_growth_rate_default;

// ============================================================================================
// ============================================================================================
void
DataArrayUtils::set_growth_rate(ConfigMap const & config_map)
{
  // a growth rate of 1.2 seems reasonable
  m_capacity_growth_rate =
    config_map.getReal("data_array", "capacity_growth_rate", KALYPSSO_NUM(1.2));

  KOKKOS_ASSERT(m_capacity_growth_rate >= KALYPSSO_NUM(1.0) &&
                "Memory capacity growth rate must be larger than one.");
}

// ============================================================================================
// ============================================================================================
void
DataArrayUtils::set_growth_rate(real_t growth_rate)
{
  m_capacity_growth_rate = growth_rate;

  KOKKOS_ASSERT(m_capacity_growth_rate >= KALYPSSO_NUM(1.0) &&
                "Memory capacity growth rate must be larger than one.");
}

} // namespace kalypsso
