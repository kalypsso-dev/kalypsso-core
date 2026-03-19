// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshMap.h
 * \brief Container for mesh data adapted from Kokkos::UnorderedMap.
 *
 */
#ifndef KALYPSSO_CORE_MESHMAP_H_
#define KALYPSSO_CORE_MESHMAP_H_

#include <kalypsso/core/kalypsso_core_base.h>

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h> // for CONNECTIVITY_PERIODIC_FALSE
// #include <kalypsso/core/AMRmesh_utils.h>
#include <kalypsso/core/brick_utils.h>
#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/OutsideQuadsInfo.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/brick_connectivity_utils.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/Kokkos_extensions.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/ComputeConformalStatus.h>
#include <kalypsso/core/ConformalFullStatus.h>
#include <kalypsso/core/ComputeConformalFullStatus.h>

#include <utility> // for std::pair

namespace kalypsso
{

template <class amr_map_t>
KOKKOS_INLINE_FUNCTION auto
key_to_value(uint64_t const &  key,
             amr_map_t const & amr_map) -> Kokkos::pair<bool, typename amr_map_t::value_type>
{
  using value_t = typename amr_map_t::value_type;

  // find key index in map
  auto key_index_in_map = amr_map.find(key);

  auto is_valid = amr_map.valid_at(key_index_in_map);

  value_t value = is_valid ? amr_map.value_at(key_index_in_map) : value_t{};
  return Kokkos::pair<bool, value_t>(is_valid, value);
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
/**
 * This class is a kokkos-based class implementing a bidirectional dictionary to map
 * a key (orchard key identifying a unique quadrant/octant of a p4est mesh), to a memory index
 * where other application-specific data can be found for this quadrant/octant.
 *
 * This class holds 2 data structures:
 * - the hashmap (using Kokkos::UnorderedMap): maps orchard keys to memory index (Morton order)
 * - an array of orchard keys (stored in the Morton order): maps index to key
 *
 * These two data structure together constitute the birectionnal dictionary.
 */
template <size_t dim, typename device_t>
class MeshMap
{
public:
  using value_t = iOct_t;
  using exec_space = typename device_t::execution_space;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using orchard_key_view_host_t = typename orchard_key_base_t<device_t>::view_host_t;

  //! type alias for a device hashmap with key=orchard_key and value=memory index
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

  //! type alias for a host mirror of amr_hashmap_t
  using amr_hashmap_host_t = typename amr_hashmap_t::HostMirror;

private:
  //! config map (input parameter)
  const ConfigMap & m_config_map;

  //! parallel environment
  const ParallelEnv & m_par_env;

  //! unordered map from orchard key to memory index
  //!
  //! Until a better idea pops out, we store the hash map here
  //! - amr_hashmap is used in UserDataRemapper but not owned / created there
  //! - amr_hashmap is used in MeshGhostsExchanger but not owned / created there
  amr_hashmap_t m_amr_hashmap;

  //! host mirror of m_amr_hashmap
  amr_hashmap_host_t m_amr_hashmap_host;

  //! orchard keys (locally owned + MPI ghosts + outside) on device
  orchard_key_view_t m_orchard_keys;

  //! orchard keys (locally owned + MPI ghosts + outside) on host
  orchard_key_view_host_t m_orchard_keys_host;

  //! orchard keys (mirror keys) on device.
  //! just to remember mirror quadrants are quadrants that are used to pack data for MPI
  //! communication send buffer and we will sent to fill MPI ghosts quadrant in other MPI processes.
  orchard_key_view_t m_mirror_orchard_keys;

  //! same as m_mirror_orchard_keys but on host
  orchard_key_view_host_t m_mirror_orchard_keys_host;

  //! for each local tree (i.e. a tree inside current MPI task), store the memory index
  //! of the first quadrant of these trees
  //! \TODO: clarify if this is really needed
  // Kokkos::View<uint32_t *, device_t> m_local_first_quad_id_per_tree;

  //! p4est connectivity sizes.
  //! brick sizes are used to transform logical coordinates into vertex space (real space)
  brick_size_t<dim> m_brick_sizes;

  //! array of bool to tell if mesh is periodic or not (one value per direction).
  Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! outside quadrants information
  OutsideQuadsInfo m_outside_quads_info;

  //! structure storing current state of AMR mesh (number of owned, ghost and outside quadrants).
  AMRMeshInfo m_amr_mesh_info;

  //! conformal face status array on device (size : locally owned quadrants + MPI quadrants)
  conformal_status_view_t<dim, device_t> m_conformal_status_view;

public:
  // =========================================================================
  // =========================================================================
  //! constructor
  MeshMap(ConfigMap const & config_map, ParallelEnv const & par_env);

  // =========================================================================
  // =========================================================================
  //! destructor
  ~MeshMap() = default;

  // =========================================================================
  // =========================================================================
  auto
  orchard_keys()
  {
    return m_orchard_keys;
  }

  // =========================================================================
  // =========================================================================
  auto
  orchard_keys() const
  {
    return m_orchard_keys;
  }

  // =========================================================================
  // =========================================================================
  auto
  orchard_keys_clone()
  {
    using ExecutionSpace = typename orchard_key_view_t::execution_space;
    return KokkosExt::clone(ExecutionSpace{}, m_orchard_keys);
  }

  // =========================================================================
  // =========================================================================
  auto
  orchard_keys_host()
  {
    return m_orchard_keys_host;
  }

  // =========================================================================
  // =========================================================================
  auto
  orchard_keys_host_clone()
  {
    using ExecutionSpace = typename orchard_key_view_host_t::execution_space;
    return KokkosExt::clone(ExecutionSpace{}, m_orchard_keys_host);
  }

  // =========================================================================
  // =========================================================================
  auto
  mirror_orchard_keys()
  {
    return m_mirror_orchard_keys;
  }

  // =========================================================================
  // =========================================================================
  auto
  mirror_orchard_keys_host()
  {
    return m_mirror_orchard_keys_host;
  }

  // =========================================================================
  // =========================================================================
  auto
  hashmap()
  {
    return m_amr_hashmap;
  }

  // =========================================================================
  // =========================================================================
  auto
  hashmap_clone()
  {
    amr_hashmap_t clone;
    clone.rehash(m_amr_hashmap.size());
    Kokkos::deep_copy(clone, m_amr_hashmap);
    return clone;
    // return KokkosExt::clone_unordered_map(m_amr_hashmap);
  }

  // =========================================================================
  // =========================================================================
  auto
  hashmap_host()
  {
    return m_amr_hashmap_host;
  }

  // =========================================================================
  // =========================================================================
  auto
  hashmap_host_clone()
  {
    amr_hashmap_host_t clone;
    clone.rehash(m_amr_hashmap_host.size());
    Kokkos::deep_copy(clone, m_amr_hashmap_host);
    return clone;
    // return KokkosExt::clone_unordered_map(m_amr_hashmap_host);
  }

  // =========================================================================
  // =========================================================================
  auto
  conformal_status()
  {
    return m_conformal_status_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  conformal_status_host()
  {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_conformal_status_view);
  }

  // =========================================================================
  // =========================================================================
  auto
  is_brick_periodic() const
  {
    return m_is_brick_periodic;
  }

private:
  // =========================================================================
  // =========================================================================
  //! create a host view of all orchard keys (owned + ghost) using current state of p4est.
  //!
  //! for each quadrant in p4est mesh (locally owned by current MPI process, as well as mpi ghost
  //! quadrants), we compute orchard key and put it in a Kokkos::View. The first local_num_quadrants
  //! items are the owned quadrant of current MPI process, then the additional
  //! ghost->ghosts.elem_count items are the orchard keys of the ghost quadrants.
  //! Important note:
  //! - if p4est brick connectivity is periodic, external border is treated as any MPI border,
  //!   nothing special to do
  //! - if p4est brick connectivity is not periodic, we chose to include a layer of ghost block
  //!   outside the domain
  //!
  //! \param[in] forest is the forest data mesh
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  //! \note m_orchard_keys_host a host view of all compute orchards keys (locally owned + MPI
  //! ghost); it must have been allocated in the calling function
  //!
  //! Be careful forest and ghost are C structs (no constness inside C API)
  //! const qualifier are commented here, just to inform the developper/user that this function
  //! is not intended to modify the mesh nor the ghost quadrants.
  //!
  //! This function assumes (and asserts) orchard_keys_host has the expect size; so memory
  //! allocation must be done in the calling function.
  //!
  //! TODO: this function should probably be made private, only update_orchard_keys should be
  //! used publicly
  void
  compute_orchard_keys_view_host(/*const*/ forest_t<dim> * forest,
                                 /*const*/ ghost_t<dim> *  ghost);

public:
  // =========================================================================
  // =========================================================================
  //! Update host and device views of ALL orchard keys (locally owned + MPI ghost + outside) using
  //! current state of p4est.
  //!
  //! Two steps:
  //! 1. update host view (\see compute_orchard_keys_view_host) using p4est API
  //! 2. upload on device
  //!
  //! \param[in] forest is the forest data mesh
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  void
  update_orchard_keys(/*const*/ forest_t<dim> * forest,
                      /*const*/ ghost_t<dim> *  ghost);

  // =========================================================================
  // =========================================================================
  //! Update device views of ALL conformal status.
  //!
  //! Take as input orchard_keys view and compute conformal status (for each owned + MPI quadrants).
  //! More precisely, ComputeConformalStatusFunctor fills kokkos view m_conformal_status_view, which
  //! a 1D array of integers (8bit in 2d and 16bit in 3D), one integer per quadrant, encoding each
  //! face conformal status.
  //! Conformal status is defined in ConformalFaceStatus.h, see struct conformal_face_status_t.
  //!
  void
  update_conformal_status();

  //! compute conformal full status
  auto
  update_conformal_full_status() -> conformal_full_status_view_t<dim, device_t>;

private:
  // =========================================================================
  // =========================================================================
  //! create a host view of all mirror quadrant's orchard keys using current state of p4est.
  //!
  //! for each mirror quadrant in p4est mesh,
  //! compute orchard key and put it in a Kokkos::View.
  //!
  //! this routine is mostly useful for performing mesh ghost exchange (MPI). See class
  //! MeshGhostsExchanger.
  //!
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  //! be careful forest and ghost are C structs (no constness inside C API)
  //! const qualifier are commented here, just to inform the developper/user that this function
  //! is not intended to modify the mesh nor the ghost quadrants.
  //!
  //! IMPORTANT NOTE: orchard key are sorted by MPI processor to send to and then by Morton order
  //!
  //! TODO: this function should probably be made private, only update_orchard_keys should be
  //! used publicly
  void
  compute_mirror_orchard_keys_view_host(/*const*/ ghost_t<dim> * ghost);

public:
  // =========================================================================
  // =========================================================================
  //! Update host and device view of (ghost) mirror quadrant's orchard keys using current state of
  //! p4est.
  //!
  //! Two steps:
  //! 1. create a host view using p4est API
  //! 2. upload on device
  //!
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  //!
  void
  update_mirror_orchard_keys(/*const*/ ghost_t<dim> * ghost);

  // =========================================================================
  // =========================================================================
  //! create both host and device views of mirror quadrant's orchard keys using current state of
  //! p4est.
  //!
  //! Two steps:
  //! 1. create a host view (\see create_orchard_keys_view_serial) using p4est API
  //! 2. upload on device
  //!
  //! \param[in] forest is the forest data mesh
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  //! \return a pair of Kokkos::View (host and device) of orchards keys (mirror quadrants only)
  // std::pair<orchard_key_view_host_t, orchard_key_view_t>
  // create_mirror_orchard_keys_views(
  //   /*const*/ forest_t<dim> * forest,
  //   /*const*/ ghost_t<dim> *  ghost,
  //   std::string               view_name)
  // {

  //   assertm(ghost != nullptr, "ghost is not valid / allocated");

  //   // memory allocation for device view
  //   orchard_key_view_t keys_view_device(Kokkos::view_alloc(Kokkos::WithoutInitializing,
  //   view_name),
  //                                       ghost->mirror_proc_offsets[m_par_env.size()]);

  //   // create view on host
  //   auto keys_view_host = Kokkos::create_mirror_view(keys_view_device);

  //   compute_mirror_orchard_keys_view_host(forest, ghost);

  //   // upload view to device
  //   Kokkos::deep_copy(keys_view_device, keys_view_host);

  //   return { keys_view_host, keys_view_device };

  // } // create_mirror_orchard_keys_views

  // =========================================================================
  // =========================================================================
  //! update hashmap (key = orchard key, value = memory index).
  //!
  //! this routine assume the orchard keys arrays are up to date.
  //!
  void
  update_hashmap(bool on_device = false);

  // =========================================================================
  // =========================================================================
  //! update orchard keys array and then update hashmap (key = orchard key, value = memory index)
  //!
  //! There are 2 slightly different implementations (TODO: benchmark which can be fastest when
  //! device is GPU).
  //!
  //! In any case, we start by updating the orchard keys arrays (host and device).
  //!
  //! if on_device is true, fill the hash map on device (using our device_t)
  //! if on_device is false, fill the hash map on host (using Kokkos::OpenMP)
  //!
  //! \param[in] forest
  //! \param[in] ghost
  //! \param[in] on_device : if true, filling hash_map is update on device and copied on host
  //!                        else updated on host first then copied on device
  //!
  void
  update_hashmap(/* const */ forest_t<dim> * forest,
                 /* const */ ghost_t<dim> *  ghost,
                 bool                        on_device = false);

  // =========================================================================
  // =========================================================================
  //! clear and fill hash table using current state of p4est.
  //! for each quadrant in p4est mesh + mpi ghost quadrants,
  //! compute orchard key and insert tuple (orchard_key, octant index)
  //! into the hash table
  //!
  //! \param[in] forest is the forest data mesh
  //! \param[in] ghost is the p4est data structure containing array of ghost quadrant
  //!
  //! be careful there are C structs (no constness inside C API)
  //! const qualifier are commented here, just to inform the developper/user that this function
  //! is not intended to modify the mesh nor the ghost quadrants.
  //!
  //! \note beware this function only consider owned and MPI ghost quadrants. Outside quadrants
  //! are not used here. Please use fill_map instead. fill_map internally first updates the
  //! orchard keys array by considering all keys (owned, ghost and outside), and then fill the map
  //! with them.
  void
  update_hashmap_serial(/*const*/ forest_t<dim> * forest, /*const*/ ghost_t<dim> * ghost);

  // =========================================================================
  // =========================================================================
  //! Pseudo check to test if current amr_mesh_info is up to date.
  bool
  is_amr_mesh_info_uptodate(/*const*/ forest_t<dim> * forest, /*const*/ ghost_t<dim> * ghost);

  // =========================================================================
  // =========================================================================
  //! compute outside quad information
  //!
  //! This routine assumes m_num_outside_quads and m_num_outside_ghosts are up to date.
  //! This is ok to call it from inside update_amr_mesh_info.
  void
  compute_outside_quad_info(/*const*/ forest_t<dim> * forest, /*const*/ ghost_t<dim> * ghost);

  // =========================================================================
  // =========================================================================
  //! provide outside quadrants information.
  OutsideQuadsInfo
  get_outside_quads_info() const
  {
    return m_outside_quads_info;
  }

  // =========================================================================
  // =========================================================================
  //! provide a valid instance of AMRMeshInfo from m_outside_quads_info.
  //!
  //! computes the number of quadrants per type (owned, ghost, outside and outside_ghost)
  void
  update_amr_mesh_info(/*const*/ forest_t<dim> * forest, /*const*/ ghost_t<dim> * ghost);

  // =========================================================================
  // =========================================================================
  //! provide outside quadrants information.
  AMRMeshInfo const &
  get_amr_mesh_info() const
  {
    return m_amr_mesh_info;
  }

  // =========================================================================
  // =========================================================================
  //! provide outside quadrants information.
  OutsideQuadsInfo &
  get_outside_quads_info()
  {
    return m_outside_quads_info;
  }

}; // class MeshMap

// explicit template instantiation
extern template class MeshMap<2, kalypsso::DefaultDevice>;
extern template class MeshMap<3, kalypsso::DefaultDevice>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
extern template class MeshMap<2, kalypsso::HostDevice>;
extern template class MeshMap<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE


} // namespace kalypsso

#endif // KALYPSSO_CORE_MESHMAP_H_
