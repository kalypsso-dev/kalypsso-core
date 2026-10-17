// SPDX-FileCopyrightText: 2026 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRManager.h
 *
 * \brief Contains the type that handles the AMR status, profiling, updates and load balancing.
 */

#ifndef KALYPSSO_CORE_AMR_MANAGER_H_
#define KALYPSSO_CORE_AMR_MANAGER_H_

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/amr_cycle_types.h>
#include <kalypsso/core/HydroParams.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#  include <kalypsso/core/MeshPartitioner.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/utils/monitoring/ProfilingManager.h>

#include <memory>

namespace kalypsso
{

namespace core
{

/**
 * \class AMRManager
 * \brief Handles the AMR status, profiling, updates and load balancing. Holds the details about
 * the mesh.
 *
 * \tparam dim The dimension of the problem (must be 2 or 3).
 * \tparam device_t On which Kokkos device to run the underlying functor.
 */
template <size_t dim, typename device_t>
class AMRManager
{
public:
  /**
   * \brief Creates the AMR mesh.
   *
   * \param par_env Parallel environment.
   * \param config_map Inputted config map.
   *
   * The profiling manager is configured with data from amr_cycle_profiling.h
   */
  AMRManager(ParallelEnv const & par_env, ConfigMap const & config_map);

  /**
   * \brief Gets the AMR mesh map in a shared pointer (used to initialize HDF5 Writer)
   *
   * \return A shared pointer to a MeshMap
   */
  std::shared_ptr<MeshMap<dim, device_t>>
  get_mesh_map()
  {
    return m_mesh_map;
  }

  AMRmesh<dim> &
  get_mesh()
  {
    return m_mesh;
  }

  /**
   * \brief Gets p4est pointer.
   */
  auto
  get_forest()
  {
    return m_mesh.forest();
  }

  /**
   * \brief Gets the number of owned octants
   *
   * \return The number of owned octants
   */
  auto
  get_nb_owned()
  {
    return m_mesh.local_num_quadrants();
  }

  /**
   * \brief Gets the number of mirror octants
   *
   * \return The number of mirror octants
   */
  auto
  get_nb_mirrors()
  {
    return m_mesh.local_num_mirrors();
  }

  /**
   * \brief Gets the number of MPI ghost octants
   *
   * \return The number of MPI ghost octants
   */
  auto
  get_nb_ghosts()
  {
    return m_mesh.local_num_ghosts();
  }

  /**
   * \brief Gets the number of local outside octants
   *
   * \return The number of local outside octants
   */
  auto
  get_nb_outside()
  {
    return m_mesh_map->get_amr_mesh_info().total_local_number_of_outside_quads();
  }

  /**
   * \brief Gets the start of local outside octants
   *
   * \return The start of local outside octants
   */
  auto
  get_first_outside()
  {
    return m_mesh_map->get_amr_mesh_info().first_outside_quad_local_id();
  }

  /**
   * \brief Gets the total number of octants (owned + MPI ghost + outside + MPI outside ghost)
   *
   * \return The total number of octants
   */
  auto
  get_nb_total()
  {
    return m_mesh_map->get_amr_mesh_info().local_num_quadrants_total();
  }

  /**
   *
   */
  auto
  get_amr_mesh_info()
  {
    return m_mesh_map->get_amr_mesh_info();
  }

  /**
   * \brief Gets the Orchard keys.
   *
   * \return The array of Orchard keys.
   */
  core::OrchardKeys<device_t>
  get_orchard_keys()
  {
    return m_mesh_map->orchard_keys();
  }

  /**
   * \brief Gets the mirror Orchard keys.
   *
   * \return The array of mirror Orchard keys.
   */
  core::OrchardKeys<device_t>
  get_mirror_orchard_keys()
  {
    return m_mesh_map->mirror_orchard_keys();
  }

  /**
   * \brief Gets a clone of Orchard keys.
   *
   * \return Array of Orchard keys.
   */
  core::OrchardKeys<device_t>
  get_orchard_keys_clone()
  {
    return m_mesh_map->orchard_keys_clone();
  }

  /**
   * \brief Gets the Orchard keys on host.
   *
   * \return The array of Orchard keys on host.
   */
  auto
  get_orchard_keys_host()
  {
    return m_mesh_map->orchard_keys_host();
  }

  /**
   * \brief Gets the AMR hashmap linking keys to indices.
   *
   * \return The AMR hashmap.
   */
  core::AMRHashmap<device_t>
  get_hashmap()
  {
    return m_mesh_map->hashmap();
  }

  /**
   * \brief Gets a AMR hashmap clone.
   *
   * \return The AMR hashmap.
   */
  core::AMRHashmap<device_t>
  get_hashmap_clone()
  {
    return m_mesh_map->hashmap_clone();
  }

  //! Get AMR flags on device
  auto
  get_amr_flags_on_device()
  {
    return m_amr_context.m_amrflags_d;
  }

  //! Get AMR flags on host
  auto
  get_amr_flags_on_host()
  {
    return m_amr_context.m_amrflags_h;
  }

  //! get conformal status view on device
  auto
  get_conformal_status()
  {
    return m_mesh_map->conformal_status();
  }

  //! get conformal status view on device
  auto
  get_conformal_status_host()
  {
    return m_mesh_map->conformal_status_host();
  }

#ifdef KALYPSSO_CORE_USE_MPI
  /**
   * \brief Gets the mesh ghosts exchanger.
   *
   * \return The mesh ghosts exchanger.
   */
  MeshGhostsExchanger<dim, real_t, device_t> &
  get_mesh_ghosts_exchanger()
  {
    return m_mesh_ghosts_exchanger;
  }
#endif // KALYPSSO_CORE_USE_MPI

  /**
   * \brief Reset AMR flags
   */
  void
  reset_amr_flags();

  /**
   * \brief Does the AMR update
   *
   * \param solver_data Solver user data to remap.
   * \param solver_data Auxiliary solver user data.
   * \param config_map Inputted config map.
   */
  void
  amr_update(DataArrayBlock<dim, real_t, device_t> & solver_data,
             DataArrayBlock<dim, real_t, device_t> & solver_data_aux,
             const ConfigMap &                       config_map);

  /**
   * \brief Recomputes the mesh from some context.
   *
   * \param amr_context The AMR context to recompute the mesh from.
   */
  void
  rebuild_from_context();

  /**
   * \brief Recomputes the mesh.
   *
   * \param reset_ghosts Recompute the ghosts
   */
  void
  rebuild(bool reset_ghosts);

  /**
   * \brief Repartition mesh
   */
  void
  repartition_mesh();

  /**
   * \brief Repartition user data
   *
   * \param solver_data Solver data to repartition (usually the conservative variables).
   * \param solver_data_aux Auxiliary solver data.
   */
  void
  repartition_userdata(DataArrayBlock<dim, real_t, device_t> & solver_data,
                       DataArrayBlock<dim, real_t, device_t> & solver_data_aux);

  /**
   * \brief Does the load balancing
   *
   * \param solver_data Solver data to repartition (usually the conservative variables).
   * \param solver_data_aux Auxiliary solver data.
   */
  void
  load_balancing(DataArrayBlock<dim, real_t, device_t> & solver_data,
                 DataArrayBlock<dim, real_t, device_t> & solver_data_aux);

  /**
   * \brief Does the load balancing and copy data
   *
   * \param solver_data Solver data to repartition (usually the conservative variables).
   * \param solver_data_aux Auxiliary solver data.
   */
  void
  load_balancing_and_copy(DataArrayBlock<dim, real_t, device_t> & solver_data,
                          DataArrayBlock<dim, real_t, device_t> & solver_data_aux);

  /**
   * \brief Print timings.
   *
   * \param[in] total_time Total time used to compute percentage.
   * \param[in] header A string to customize printing.
   */
  void
  print_timings(double total_time, std::string header)
  {
    m_profiling_mgr.print_timings(total_time, header);
  }

private:
  //! Kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  /**
   * \brief Remaps data from hashmap
   *
   * \param solver_data The data to remap
   * \param solver_data_aux Auxiliary data used during remap
   * \param old_hashmap The previous AMR hashmap
   * \param old_keys The old Orchard key array
   * \param config_map Inputted config map.
   */
  void
  remap_from_hashmap(DataArrayBlock<dim, real_t, device_t> & solver_data,
                     DataArrayBlock<dim, real_t, device_t> & solver_data_aux,
                     core::AMRHashmap<device_t>              old_hashmap,
                     core::OrchardKeys<device_t>             old_keys,
                     const ConfigMap &                       config_map);

  //! AMR data
  AMRmesh<dim> m_mesh;

  //! Bidirectional map orchard_key <-> morton id
  std::shared_ptr<MeshMap<dim, device_t>> m_mesh_map;

  //! Storage for AMR refine/coarsen flags and actual AMR mesh adaptation by calling p4est APIs
  AMRContext<dim, device_t> m_amr_context;

#ifdef KALYPSSO_CORE_USE_MPI
  //! Mesh exchanger
  MeshGhostsExchanger<dim, real_t, device_t> m_mesh_ghosts_exchanger;

  //! Mesh partitioner
  MeshPartitioner<dim, device_t> m_mesh_partitioner;
#endif // KALYPSSO_CORE_USE_MPI

  //! Profiling manager
  ProfilingManager m_profiling_mgr;

}; // class AMRManager

extern template class AMRManager<2, kalypsso::DefaultDevice>;
extern template class AMRManager<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMR_MANAGER_H_
