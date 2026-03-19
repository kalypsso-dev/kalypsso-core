// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/problems/init_cond_utils.h>

#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{
namespace core
{

InitConditionsIndicator
get_init_indicator(ConfigMap const & config_map)
{
  const auto indicator_name =
    config_map.getString("amr", "refine_criterion_at_init", "SAME_AS_REGULAR_DYNAMICS");
  auto maybe_value = InitConditionsIndicator::_from_string_nothrow(indicator_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return InitConditionsIndicator::SAME_AS_REGULAR_DYNAMICS;
}

} // namespace core
} // namespace kalypsso
