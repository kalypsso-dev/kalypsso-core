// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MultiFluidShockTubeParams.h
 *
 * All parameters for a multi-fluid shock tube problem.
 *
 * Each fluid has its own equation of state (e.g. stiffened gas, ...)
 */
#ifndef KALYPSSO_CORE_PROBLEMS_MULTIFLUIDSHOCKTUBE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_MULTIFLUIDSHOCKTUBE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * MultiFluidShockTube test parameters.
 *
 */
template <typename device_t>
struct MultiFluidShockTubeParams
{
  //! number of regions
  int nb_regions;

  //! discontinuity location
  Kokkos::View<real_t *, device_t> xd;

  MultiFluidShockTubeParams(ConfigMap const & config_map)
    : nb_regions(config_map.getInteger("multi_fluid_shock_tube", "nb_regions", 2))
  {

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
    // const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    // const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    // const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    // const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    // const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    // const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    // discontinuity positions (a vector of nb_regions-1 values)
    auto xd_h = config_map.getRealVector(
      "multi_fluid_shock_tube", "xd", std::vector<real_t>{ (xmin + xmax) / KALYPSSO_NUM(2.0) });
    assertm(xd_h.size() == static_cast<size_t>(nb_regions - 1),
            "[MultiFluidShockTubeParams] wrong size.");
    xd = to_view<real_t, typename device_t::memory_space>(xd_h);
  }

}; // struct MultiFluidShockTubeParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_MULTIFLUIDSHOCKTUBE_PARAMS_H_
