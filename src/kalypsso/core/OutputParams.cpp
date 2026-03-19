// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file OutputParams.cpp
 */
#include "OutputParams.h"

namespace kalypsso
{

// =======================================================
// =======================================================
OutputParams::OutputParams(ConfigMap const & config_map)
  : nOutput(config_map.getInteger("run", "noutput", 100))
  , enable_output(nOutput == 0 ? false : true)
  , output_policy(get_output_policy(config_map))
  , hdf5_enabled(config_map.getBool("output", "hdf5_enabled", false))
  , exabrick_enabled(config_map.getBool("output", "exabrick_enabled", false))
  , debug_output(config_map.getBool("output", "debug", false))
  , output_times(config_map.getRealVector("run", "output_times", std::vector<real_t>{}))
{
  if (output_policy == +OUTPUT_POLICY::PREDEFINED)
  {
    nOutput = static_cast<int>(output_times.size());

    // check if output times are in increasing order
    bool increasing_order = true;
    for (size_t i = 0; i < output_times.size() - 1; ++i)
    {
      if (output_times[i] >= output_times[i + 1])
      {
        increasing_order = false;
        break;
      }
    }
    if (!increasing_order)
    {
      Kokkos::abort("Predefined output times must be given in strictly increasing order.");
    }

    // make sure initial times is not in the list
    const auto t_begin = config_map.getReal("run", "tBegin", KALYPSSO_NUM(0.0));
    if (output_times.size() > 0 and fabs(output_times[0] - t_begin) < KALYPSSO_NUM(1e-14))
    {
      Kokkos::abort("Remove initial time from predefined output times.");
    }
  }
} // OutputParams::OutputParams

} // namespace kalypsso
