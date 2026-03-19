// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDSettings.cpp
 */
#include "MHDSettings.h"

namespace kalypsso
{

// =======================================================
// =======================================================
MHDSettings::MHDSettings(ConfigMap const & config_map)
  : hydro(config_map)
  , mag_field_split_enabled(config_map.getBool("mhd", "mag_field_split_enabled", false))
  , correction_boris_enabled(config_map.getBool("mhd", "correction_boris_enabled", false))
  , cboris(config_map.getReal("mhd", "cboris", KALYPSSO_NUM(1.0)))
  , correction_entropy_enabled(config_map.getBool("mhd", "correction_entropy_enabled", false))
  , small_beta(config_map.getReal("mhd", "small_beta", KALYPSSO_NUM(1e-3)))
  , large_alfven(config_map.getReal("mhd", "large_alfven", KALYPSSO_NUM(10.0)))
{}

// =======================================================
// =======================================================
void
MHDSettings::print() const
{
  hydro.print();

  KALYPSSO_INFO("mag_field_split_enabled        : {}", mag_field_split_enabled);
  KALYPSSO_INFO("correction_boris_enabled       : {}", correction_boris_enabled);
  KALYPSSO_INFO("cboris (MHD-CC only)           : {}", cboris);
  KALYPSSO_INFO("correction_entropy_enabled     : {}", correction_entropy_enabled);
  KALYPSSO_INFO("small_beta                     : {}", small_beta);
  KALYPSSO_INFO("large_alfven                   : {}", large_alfven);

} // MHDSettings::print

} // namespace kalypsso
