// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroParams.cpp
 */
#include "HydroParams.h"

#include <cstdlib> // for exit
#include <cstdio>  // for fprintf
#include <cstring> // for strcmp
#include <iostream>

#include <kalypsso/utils/config/inih/ini.h> // our INI file reader
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/enums.h>

namespace kalypsso
{

// =======================================================
// =======================================================
size_t
get_dim(ConfigMap const & config_map)
{
  // retrieve solver name from settings
  const auto dimension = static_cast<size_t>(config_map.getInteger("run", "dimension", 0));

  if (dimension != 2 and dimension != 3)
  {
    Kokkos::abort("dimension must be 2 or 3 (and nothing else) !");
  }

  return dimension;
} // get_dim

// =======================================================
// =======================================================
HydroParams::HydroParams(ConfigMap const & config_map)
  : nStepmax(config_map.getInteger("run", "nstepmax", 1000))
  , tEnd(config_map.getReal("run", "tend", KALYPSSO_NUM(0.0)))
  , nOutput(config_map.getInteger("run", "noutput", 100))
  , enableOutput(nOutput == 0 ? false : true)
  , nlog(config_map.getInteger("run", "nlog", 10))
  , dimType(static_cast<size_t>(config_map.getInteger("run", "dimension", 0)))
  , xmin(config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0)))
  , xmax(1.0)
  , ymin(config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0)))
  , ymax(1.0)
  , zmin(config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0)))
  , zmax(1.0)
  , level_min(config_map.getInteger("amr", "level_min", 5))
  , level_max(config_map.getInteger("amr", "level_max", 10))
  , amr_cycle_enabled(false)
  , output_hdf5_enabled(config_map.getBool("output", "hdf5_enabled", false))
  , output_exabrick_enabled(config_map.getBool("output", "exabrick_enabled", false))
  , debug_output(config_map.getBool("output", "debug", false))
  , updateType(UPDATE_CONSERVATIVE_SUM)
  , replicated_init_cond(config_map.getBool("hydro", "replicated_init_cond", false))
{
  if (dimType != 2 and dimType != 3)
  {
    // we should probably abort
    std::cerr << "dimension is not valid (can only be 2 or 3); given value is " << dimType << "\n";
    Kokkos::abort("Invalid dimension");
  }

  setup(config_map);

} // HydroParams::HydroParams

// =======================================================
// =======================================================
void
HydroParams::setup(ConfigMap const & config_map)
{

  /* initialize MESH parameters */

  const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
  const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
  const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

  const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

  xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
  ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
  zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

  // default value for amr_cycle_enabled
  bool amr_cycle_enabled_default = (level_min != level_max);

  // we can overwrite amr_cycle_enabled; e.g.
  // we can chose level_min != level_max for initial condition
  // but then switch-off amr cycle.
  amr_cycle_enabled = config_map.getBool("amr", "amr_cycle_enabled", amr_cycle_enabled_default);

  std::string utype = config_map.getString("hydro", "updateType", "conservative_sum");
  if (utype == "conservative_sum")
    updateType = UPDATE_CONSERVATIVE_SUM;
  else
    updateType = UPDATE_NON_CONSERVATIVE;

} // HydroParams::setup

// =======================================================
// =======================================================
void
HydroParams::print() const
{

  KALYPSSO_INFO("##########################");
  KALYPSSO_INFO("Simulation run parameters:");
  KALYPSSO_INFO("##########################");
  KALYPSSO_INFO("nStepmax   : {}", nStepmax);
  KALYPSSO_INFO("tEnd       : {}", tEnd);
  KALYPSSO_INFO("nOutput    : {}", nOutput);
  KALYPSSO_INFO("update type: {}", updateType);
  KALYPSSO_INFO("level_min  : {}", level_min);
  KALYPSSO_INFO("level_max  : {}", level_max);
  KALYPSSO_INFO("amr cycle enabled : {}", amr_cycle_enabled);
  KALYPSSO_INFO("##########################");

} // HydroParams::print

} // namespace kalypsso
