// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRMeshMonitoring.h
 */
#ifndef KALYPSSO_CORE_AMRMESHMONITORING_H_
#define KALYPSSO_CORE_AMRMESHMONITORING_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t and amr_map_t
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

/**
 * A small class to compute, gather and print monitoring information about AMR mesh.
 */
template <size_t dim, typename device_t>
class AMRMeshMonitoring
{
public:
  using exec_space = typename device_t::execution_space;

  using level_histo_view_t = Kokkos::View<int *, device_t>;
  using MeshMap_t = MeshMap<dim, device_t>;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  AMRMeshMonitoring(ConfigMap const & config_map)
    : m_level_min(config_map.getInteger("amr", "level_min", 0))
    , m_level_max(config_map.getInteger("amr", "level_max", 0))
    , m_num_levels(m_level_max - m_level_min + 1)
    , m_num_trees(config_map.getInteger("p4est_connectivity", "nbrick_x", 1) *
                  config_map.getInteger("p4est_connectivity", "nbrick_y", 1) *
                  config_map.getInteger("p4est_connectivity", "nbrick_z", 1))
  {}

  ~AMRMeshMonitoring() = default;

  // ============================================================================================
  // ============================================================================================
  void
  print_level_info(ParallelEnv const & par_env, MeshMap<dim, device_t> const & mesh_map);

  // ============================================================================================
  // ============================================================================================
  auto
  compute_level_histogram([[maybe_unused]] ParallelEnv const & par_env,
                          MeshMap<dim, device_t> const &       mesh_map) -> level_histo_view_t;

private:
  int m_level_min;
  int m_level_max;
  int m_num_levels;
  int m_num_trees;

}; // class AMRMeshMonitoring

// explicit template instantiation
extern template class AMRMeshMonitoring<2, DefaultDevice>;
extern template class AMRMeshMonitoring<3, DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class AMRMeshMonitoring<2, HostDevice>;
extern template class AMRMeshMonitoring<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRMESHMONITORING_H_
