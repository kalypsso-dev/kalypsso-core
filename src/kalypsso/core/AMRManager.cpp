// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRManager.cpp
 *
 * \brief Contains the definition of AMRManager.
 */

#include <kalypsso/core/AMRManager.h>
#include <kalypsso/core/amr_cycle_profiling.h>

#include <kalypsso/core/UserDataRemapper.h>
#include <kalypsso/core/ComputeRefineFlags.h>
#include <kalypsso/core/config_utils.h>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
AMRManager<dim, device_t>::AMRManager(ParallelEnv const & par_env, ConfigMap const & config_map)
  : m_mesh(par_env, config_map)
  , m_mesh_map(std::make_shared<MeshMap<dim, device_t>>(config_map, par_env))
  , m_amr_context(0)
#ifdef KALYPSSO_CORE_USE_MPI
  , m_mesh_ghosts_exchanger(config_map, par_env, m_mesh, *m_mesh_map)
  , m_mesh_partitioner(config_map, par_env)
#endif // KALYPSSO_CORE_USE_MPI
  , m_profiling_mgr(par_env)
{
  rebuild(true);
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::reset_amr_flags()
{
  m_amr_context.resize_amrflags(get_nb_owned());
  m_amr_context.reset_amrflags();
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::rebuild_from_context()
{
  const auto changed = m_amr_context.adapt_mesh(m_mesh.forest());
  rebuild(changed);
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::rebuild(bool reset_ghosts)
{
  KALYPSSO_PROFILING_REGION_HOST(m_profiling_mgr, AMR_CYCLE_REBUILD);

  if (reset_ghosts)
  {
    m_mesh.reset_ghost();
  }
  m_mesh_map->compute_outside_quad_info(m_mesh.forest(), m_mesh.ghost());
  m_mesh_map->update_hashmap(m_mesh.forest(), m_mesh.ghost(), true);
  m_mesh_map->update_mirror_orchard_keys(m_mesh.ghost());
  m_mesh_map->update_amr_mesh_info(m_mesh.forest(), m_mesh.ghost());
  m_mesh_map->update_orchard_keys(m_mesh.forest(), m_mesh.ghost());
  m_mesh_map->update_conformal_status();
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::repartition_mesh()
{
#ifdef KALYPSSO_CORE_USE_MPI
  m_mesh_partitioner.partition_mesh(m_mesh.forest());
#endif // KALYPSSO_CORE_USE_MPI
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::repartition_userdata(
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data,
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data_aux)
{
#ifdef KALYPSSO_CORE_USE_MPI
  m_mesh_partitioner.repartition_userdata(solver_data, solver_data_aux);
#endif // KALYPSSO_CORE_USE_MPI
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::repartition_userdata(
  [[maybe_unused]] FaceDataArrayBlock<dim, real_t, device_t> & solver_data,
  [[maybe_unused]] FaceDataArrayBlock<dim, real_t, device_t> & solver_data_aux)
{
#ifdef KALYPSSO_CORE_USE_MPI
  m_mesh_partitioner.repartition_userdata(solver_data, solver_data_aux);
#endif // KALYPSSO_CORE_USE_MPI
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::load_balancing(
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data,
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data_aux)
{
#ifdef KALYPSSO_CORE_USE_MPI
  repartition_mesh();
  repartition_userdata(solver_data, solver_data_aux);
  rebuild(true);
  solver_data_aux.resize(get_nb_total());
  my_swap(solver_data, solver_data_aux);
#endif // KALYPSSO_CORE_USE_MPI
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
AMRManager<dim, device_t>::load_balancing_and_copy(
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data,
  [[maybe_unused]] DataArrayBlock<dim, real_t, device_t> & solver_data_aux)
{
  load_balancing(solver_data, solver_data_aux);
  Kokkos::deep_copy(solver_data_aux.logical_view(), solver_data.logical_view());
}

// ================================================================================================
// ================================================================================================
template class AMRManager<2, kalypsso::DefaultDevice>;
template class AMRManager<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
