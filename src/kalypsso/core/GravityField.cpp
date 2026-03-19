// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/GravityField.h>

namespace kalypsso
{

// ================================================================================
// ================================================================================
GravityType
get_gravity_type(ConfigMap const & config_map)
{
  auto gravityTypeStr = config_map.getString("gravity", "gravityType", "undefined");
  if (gravityTypeStr == "uniform")
    return GravityType::UNIFORM;
  else
    return GravityType::UNDEFINED;
}

} // namespace kalypsso
