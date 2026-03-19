// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file OutputParams.h
 */
#ifndef KALYPSSO_CORE_OUTPUT_PARAMS_H_
#define KALYPSSO_CORE_OUTPUT_PARAMS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

#include <vector>
#include <../../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
/**
 * Enumerate types of output policies.
 *
 * - EQUISPACED: output are equispaced in time with time interval equal to (t_end-t_begin)/nOutput
 * - PREDEFINED: times at which an output is requested are specified by config parameter "run"/"output_times"
 */
BETTER_ENUM(OUTPUT_POLICY, int,
            EQUISPACED,
            PREDEFINED)
// clang-format on

/**
 * Read output policy from input parameters file.
 */
inline OUTPUT_POLICY
get_output_policy(ConfigMap const & config_map)
{
  auto output_policy_name = config_map.getString("run", "output_policy", "EQUISPACED");
  auto maybe_value = OUTPUT_POLICY::_from_string_nothrow(output_policy_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return OUTPUT_POLICY::EQUISPACED;
}

/**
 * Output parameters.
 */
struct OutputParams
{
  //! number of outputs.
  //! - nOutput<0  : output at every time step iteration
  //! - nOutput=0  : no output at all
  //! - nOutput>0  : one output every Delta_t = (t_end - t_beg)/nOutput
  int nOutput;

  //! enable output file write.
  bool enable_output;

  //! output policy
  OUTPUT_POLICY output_policy;

  //! enable HDF5 output file format.
  bool hdf5_enabled;

  //! enable exabrick output file format.
  bool exabrick_enabled;

  //! more verbose output
  bool debug_output;

  //! list of additional times where output is requested
  std::vector<real_t> output_times;

  //! constructor
  OutputParams(ConfigMap const & config_map);

}; // struct OutputParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_OUTPUT_PARAMS_H_
