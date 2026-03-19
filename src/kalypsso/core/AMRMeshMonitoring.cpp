// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRMeshMonitoring.cpp
 */
#include <kalypsso/core/AMRMeshMonitoring.h>

namespace kalypsso
{

// ============================================================================================
// ============================================================================================
template <size_t dim, typename device_t>
void
AMRMeshMonitoring<dim, device_t>::print_level_info(ParallelEnv const &            par_env,
                                                   MeshMap<dim, device_t> const & mesh_map)
{
  // all MPI procesus computes its local histogram
  const auto level_histo = compute_level_histogram(par_env, mesh_map);

  const auto level_histo_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, level_histo);

  if (par_env.rank() == 0)
  {
    // compute total number of keys
    int64_t total_num_keys = 0;
    for (auto level = m_level_min; level <= m_level_max; ++level)
    {
      total_num_keys += level_histo_host(level - m_level_min);
    }

    // total number of keys if the mesh was uniformly refined at max_level
    [[maybe_unused]] int64_t total_num_keys_at_max_level_uniform =
      m_num_trees * (1 << (static_cast<int>(dim) * m_level_max));

    KALYPSSO_INFO("AMR level histogram:");
    for (auto level = m_level_min; level <= m_level_max; ++level)
    {
      const auto num_keys = level_histo_host(level - m_level_min);
      const int  scaling = 1 << (static_cast<int>(dim) * (m_level_max - level));
      const auto num_keys_at_max_level = num_keys * scaling;


      KALYPSSO_INFO("level {:3d}: {:10d}/{:10d} ({: 6.2f}%) => {: 6.2f}% of volume ({:10d}/{:10d})",
                    level,
                    num_keys,
                    total_num_keys,
                    (100.0 * num_keys) / static_cast<double>(total_num_keys),
                    (100.0 * num_keys_at_max_level) /
                      static_cast<double>(total_num_keys_at_max_level_uniform),
                    num_keys_at_max_level,
                    total_num_keys_at_max_level_uniform);
    }

    // compute sparsity index as the ratio of the total number of AMR keys divided
    // by the number of keys obtained if the mesh was fully Cartesian at level=level_max
    KALYPSSO_INFO("Sparsity index is the ratio of total number of AMR blocks ({}) over",
                  total_num_keys);
    KALYPSSO_INFO("the number of blocks if the whole mesh was at level max ({}).",
                  total_num_keys_at_max_level_uniform);
    KALYPSSO_INFO("Sparsity index : {: 3.2f}%",
                  100.0 * static_cast<double>(total_num_keys) /
                    static_cast<double>(total_num_keys_at_max_level_uniform));
  }

} // AMRMeshMonitoring<dim, device_t>::print_level_info

// ============================================================================================
// ============================================================================================
template <size_t dim, typename device_t>
auto
AMRMeshMonitoring<dim, device_t>::compute_level_histogram(
  [[maybe_unused]] ParallelEnv const & par_env,
  MeshMap<dim, device_t> const &       mesh_map) -> level_histo_view_t
{

  const auto num_octants = mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto orchard_keys = mesh_map.orchard_keys();
  auto       level_histo = level_histo_view_t("level_histo", static_cast<size_t>(m_num_levels));
  Kokkos::deep_copy(level_histo, 0);

  const auto                  level_min = m_level_min;
  [[maybe_unused]] const auto level_max = m_level_max;

  Kokkos::parallel_for(
    "compute_level_histogram",
    Kokkos::RangePolicy<exec_space>(0, num_octants),
    KOKKOS_LAMBDA(const int iOct) {
      // get AMR level
      const auto level = orchard_key_t<dim>::level(orchard_keys(iOct));
      KOKKOS_ASSERT(level >= level_min and level <= level_max and
                    "level wrong value: not in valid range.");

      Kokkos::atomic_add(&level_histo(level - level_min), 1);
    });

// gather information from all MPI processes
#ifdef KALYPSSO_CORE_USE_MPI
  auto level_histo_global =
    level_histo_view_t("level_histo_global", static_cast<size_t>(m_num_levels));
  par_env.comm().MPI_Reduce<MpiComm::SUM>(
    level_histo.data(), level_histo_global.data(), m_num_levels, 0);

  return level_histo_global;
#else
  return level_histo;
#endif // KALYPSSO_CORE_USE_MPI

} // AMRMeshMonitoring<dim, device_t>::compute_level_histogram

// explicit template instantiation
template class AMRMeshMonitoring<2, DefaultDevice>;
template class AMRMeshMonitoring<3, DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class AMRMeshMonitoring<2, HostDevice>;
template class AMRMeshMonitoring<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
