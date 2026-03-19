// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshGhostsExchanger.h
 */
#ifndef KALYPSSO_CORE_MESHGHOSTSEXCHANGER_H_
#define KALYPSSO_CORE_MESHGHOSTSEXCHANGER_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t and amr_map_t
#include <kalypsso/core/orchard_key_base.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/core/kalypsso_comm_config.h>

namespace kalypsso
{

/**
 * \class MeshGhostsExchanger
 *
 * p4est vocabulary:
 * - a mirror quadrant is a locally owned quadrant that is a ghost for another quadrant in another
 * MPI process. In other words, a mirror quadrant "touches" a MPI border between 2 MPI processes.
 * - a ghost quadrant is remote quadrant which data are locally copied into current MPI process
 *
 * Main functions:
 *
 * - make sure p4est_ghost is up to date
 * - make the ghost array is properly allocated/sized
 * - pack data from mirror quadrant into send buffer
 * - prepare receive buffer sized upon the number of ghost quadrant
 * - perform all MPI comm (either a collective version using MPI_Alltoallv or a
 *   point-to-version)
 * - unpack the receive buffer into ghost data array
 *
 * Implementation note:
 * we some follow original `p4est_ghost_exchange_custom_begin` but packing / unpacking is done on
 * device, and we require our MPI implementation to be device-aware (i.e. cuda-aware or hip-aware
 * when using GPU's).
 *
 * \tparam dim is dimension (integer: 2 or 3)
 * \tparam T is the type of each element of the userdata exchanged (most of the time real_t), used
 * to allocate send/recv buffers
 * \tparam device_t is a kokkos device class (e.g. Kokkos::CudaSpace::device_type)
 *
 */
template <size_t dim, typename T, typename device_t>
class MeshGhostsExchanger
{

public:
  using exec_space = typename device_t::execution_space;

  //! type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  using forest_t = typename p4est_t::forest_t;
  using ghost_t = typename p4est_t::ghost_t;

  using DataArrayLeafAoS_t = DataArrayLeafAoS<T, device_t>;

  using DataArrayLeaf_t = DataArrayLeaf<T, device_t>;

  using DataArrayLeafUnmanaged_t = DataArrayLeafUnmanaged<T, device_t>;

  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;
  using DataArrayBlockUnmanaged_t = DataArrayBlock<dim, T, device_t, StorageType::UNMANAGED>;

  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, T, device_t>;

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, T, device_t>;
  using FaceFlatArray_t = typename FaceDataArrayBlock_t::FaceFlatArray_t;
  using FaceFlatArrayUnmanaged_t = typename FaceDataArrayBlock_t::FaceFlatArrayUnmanaged_t;

  using MaterialPresenceView_t = MaterialPresenceView<device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, T, device_t>;

  using comm_buffer_t = Kokkos::View<T *, device_t>;
  using comm_buffer_unmanaged_t =
    Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

  // =========================================================================
  // =========================================================================
  //! constructor
  MeshGhostsExchanger(const ConfigMap &        config_map,
                      const ParallelEnv &      par_env,
                      AMRmesh<dim> &           amr_mesh,
                      MeshMap<dim, device_t> & mesh_map)
    : m_config_map(config_map)
    , m_par_env(par_env)
    , m_amr_mesh(amr_mesh)
    , m_mesh_map(mesh_map)
    , m_send(nullptr, 0)
    , m_recv(nullptr, 0)
    , m_storage("MeshGhostsExchanger::storage", 0)
    , m_data_size_per_quad(0)
  {}

  // =========================================================================
  // =========================================================================
  //! destructor
  virtual ~MeshGhostsExchanger() = default;

  // =========================================================================
  // =========================================================================
  //! resize device buffer for performing ghost data exchange.
  //!
  //! \param[in] data_size_per_quad is the number of element per block / leaf (i.e. number of cells
  //!            times number of variables) to transfer
  virtual void
  resize(size_t data_size_per_quad);

  // =========================================================================
  // =========================================================================
  /**
   * return amount of allocated memory in bytes
   */
  virtual auto
  allocated_size_in_bytes() const -> size_t;

  // =========================================================================
  // =========================================================================
  //! pack mirror data into send buffer (device function) - leaf version.
  //!
  //! implementation use array of orchard key (mirror quads) to address userdata and copy data into
  //! send buffer.
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts>elem_count)
  //! \param[in] array or orchard keys of mirror quadrant
  //!
  //! IMPORTANT NOTE: remember that orchard keys in mirror array are sorted first by MPI processor
  //! to send to and secondly by Morton order.
  //!
  //! make sure that MeshMap object is up-to-date before using this; we need the unordered map
  //! (orchard keys, index) to be valid, i.e. calling MeshMap::fill_map is necessary after any mesh
  //! changes (p4est_balance, p4est_partition)
  void
  pack_mirror_data(DataArrayLeaf_t userdata_leaf, orchard_key_view_t mirror_keys_device);

  // =========================================================================
  // =========================================================================
  //! pack mirror data into send buffer (device function) - block (cell-center) version.
  //!
  //! implementation use array of orchard key (mirror quads) to address userdata and copy data into
  //! send buffer.
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts>elem_count)
  //! \param[in] array or orchard keys of mirror quadrant
  //! \param[in] amr_hashmap is an unordered map (key=orchard_key, value=index) of all local
  //! quadrants (owned+ghosts), and please remember that mirror quadrants are locally owned
  //! quadrants)
  //!
  //! IMPORTANT NOTE: remember that orchard keys in mirror array are sorted first by MPI processor
  //! to send to and secondly by Morton order.
  //!
  //! make sure that MeshMap object is up-to-date before using this; we need the unordered map
  //! (orchard keys, index) to be valid, i.e. calling MeshMap::fill_map is necessary after any mesh
  //! changes (p4est_balance, p4est_partition)
  void
  pack_mirror_data(DataArrayBlock_t userdata_block, orchard_key_view_t mirror_keys_device);

  // =========================================================================
  // =========================================================================
  //! Does the same thing as pack_mirror_data but for multi variable data array blocks
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts>elem_count)
  //! \param[in] num_block_mirrors number of mirror blocks
  //! \param[in] array or orchard keys of mirror quadrant
  //! \param[in] mirror_offsets mirror offset array on device
  void
  pack_mirror_data_multi_var(DataArrayBlockMultiVar_t           userdata_block,
                             int32_t                            num_block_mirrors,
                             orchard_key_view_t                 mirror_keys_device,
                             Kokkos::View<uint32_t *, device_t> mirror_offsets);

  // =========================================================================
  // =========================================================================
  //! pack mirror data into send buffer (device function) - block (face-center) version.
  //!
  //! implementation use array of orchard key (mirror quads) to address userdata and copy data into
  //! send buffer.
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts>elem_count)
  //! \param[in] array or orchard keys of mirror quadrant
  //! \param[in] amr_hashmap is an unordered map (key=orchard_key, value=index) of all local
  //! quadrants (owned+ghosts), and please remember that mirror quadrants are locally owned
  //! quadrants)
  //!
  //! IMPORTANT NOTE: remember that orchard keys in mirror array are sorted first by MPI processor
  //! to send to and secondly by Morton order.
  //!
  //! make sure that MeshMap object is up-to-date before using this; we need the unordered map
  //! (orchard keys, index) to be valid, i.e. calling MeshMap::fill_map is necessary after any mesh
  //! changes (p4est_balance, p4est_partition)
  void
  pack_mirror_data(FaceDataArrayBlock_t face_userdata, orchard_key_view_t mirror_keys_device);

  // =========================================================================
  // =========================================================================
  //! unpack received data (recv buffer) into ghost data.
  //!
  //! we will fill all ghost quadrant data, i.e. from index forest->local_num_quad
  //! to index forest->local_num_quadrants + ghost->ghosts>elem_count
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts->elem_count)
  //!
  void
  unpack_ghost_data(DataArrayLeaf_t userdata_leaf);

  // =========================================================================
  // =========================================================================
  //! unpack received data (recv buffer) into ghost data (cell-center block).
  //!
  //! we will fill all ghost quadrant data, i.e. from index forest->local_num_quad
  //! to index forest->local_num_quadrants + ghost->ghosts>elem_count
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts->elem_count)
  //!
  void
  unpack_ghost_data(DataArrayBlock_t userdata_block);

  // =========================================================================
  // =========================================================================
  //! Does the same thing as unpack_mirror_data but for multi variable data array blocks
  //!
  //! \param[in] num_block_ghosts number of owned blocks
  //! \param[in] num_block_ghosts number of ghosts blocks
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts->elem_count)
  //!
  void
  unpack_ghost_data_multi_var(DataArrayBlockMultiVar_t userdata_block,
                              int32_t                  num_block_owned,
                              int32_t                  num_block_ghosts);

  // =========================================================================
  // =========================================================================
  //! unpack received data (recv buffer) into ghost data (face-center block).
  //!
  //! we will fill all ghost quadrant data, i.e. from index forest->local_num_quad
  //! to index forest->local_num_quadrants + ghost->ghosts>elem_count
  //!
  //! \param[in] userdata is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts->elem_count)
  //!
  void
  unpack_ghost_data(FaceDataArrayBlock_t face_userdata);

  // =========================================================================
  // =========================================================================
  //! actually perform MPI comm to send and receive ghost quadrant userdata
  void
  do_mpi_send_recv();

  // =========================================================================
  // =========================================================================
  //! actually perform MPI comm to send and receive ghost quadrant userdata with multi var
  //!
  //! \param[in] ghost_offsets Host view with the block offsets in m_recv
  //! \param[in] mirror_offsets Host view with the block offsets in m_send
  void
  do_mpi_send_recv_multi_var(const Kokkos::View<uint32_t *, HostDevice> & ghost_offsets,
                             const Kokkos::View<uint32_t *, HostDevice> & mirror_offsets);

  // =========================================================================
  // =========================================================================
  //! Perform MPI comm to send mirror data and receive in ghost quadrant userdata
  //! in place in input buffer
  //!
  //! Exact same communication pattern as in do_mpi_send_recv, but doing send/recv inplace.
  //!
  //! \param[in,out] userdata_ghosted_block is a userdata ghosted block array (i.e. one value per
  //!                cell); must be sized upon the sum of number of mirror quads and number of
  //!                ghost quads.
  void
  do_mpi_send_recv_inplace(DataArrayGhostedBlock_t userdata_ghosted_block);

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_leaf is a uaerdata array (at leaf, i.e. one value per leaf octant); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  //! \param[in] amr_hashmap_device is a device Kokkos::UnorderedMap of keys,value where value is
  //!            the memory index to find userdata corresponding to a given key
  //! \param[in] do_allocate is a boolean to tell if we want to allocate internal communication
  //!            buffers (can be set to false, if you know that re-allocation is not needed)
  void
  exchange(DataArrayLeaf_t userdata_leaf);

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_block is a userdata block array (i.e. one value per cell); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  //! \param[in] amr_hashmap_device is a device Kokkos::UnorderedMap of keys,value where value is
  //!            the memory index to find userdata corresponding to a given key
  //! \param[in] do_allocate is a boolean to tell if we want to allocate internal communication
  //!            buffers (can be set to false, if you know that re-allocation is not needed)
  void
  exchange(DataArrayBlock_t userdata_block);

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_block is a userdata block array (i.e. one value per face); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  //! \param[in] amr_hashmap_device is a device Kokkos::UnorderedMap of keys,value where value is
  //!            the memory index to find userdata corresponding to a given key
  //! \param[in] do_allocate is a boolean to tell if we want to allocate internal communication
  //!            buffers (can be set to false, if you know that re-allocation is not needed)
  void
  exchange(FaceDataArrayBlock_t userdata_block_face);

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_block is a userdata block array (i.e. one value per cell); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  void
  exchange_multi_var(DataArrayBlockMultiVar_t userdata_block);

  // =========================================================================
  // =========================================================================
  //! Exchange mirror/ghost without packing/unpacking, inplace.
  //!
  //! In this function we don't use send/recv buffer; the input userdata is directly used to
  //! send/recv data inplace. So to be clear, in this function the userdata in assumed to be sized
  //! upon the number of mirror quads + number of ghost quad. The first part of the array (mirror
  //! quad userdata) will be sent directly, and the second part of the array (ghosts quadrant) will
  //! be used to receive data in place.
  //!
  //! \param[in,out] userdata_ghosted_block is a userdata ghosted block array (i.e. one value per
  //!                cell); must be sized upon the sum of number of mirror quads and number of
  //!                ghost quads.
  void
  exchange_inplace(DataArrayGhostedBlock_t userdata_ghosted_block)
  {

    // make sure m_send buffer was allocated with the right size
    assertm(static_cast<size_t>(m_amr_mesh.local_num_mirrors() + m_amr_mesh.local_num_ghosts()) ==
              static_cast<size_t>(userdata_ghosted_block.num_quadrants()),
            "[MeshGhostsExchanger::exchange_inplace] input data has wrong size");

    do_mpi_send_recv_inplace(userdata_ghosted_block);

  } // exchange_inplace

protected:
  //! config map (input parameter)
  const ConfigMap & m_config_map;

  //! parallel environment
  const ParallelEnv & m_par_env;

  //! AMRmesh reference object, mostly for accessing p4est ghost object
  AMRmesh<dim> & m_amr_mesh;

  //! a MeshMap object so that we can extract the orchard keys as a kokkos view or unordered map
  MeshMap<dim, device_t> & m_mesh_map;

  //! send buffers (subview of m_storage)
  comm_buffer_unmanaged_t m_send;

  //! recv buffers (subview of m_storage)
  comm_buffer_unmanaged_t m_recv;

  //! storage buffer where send/recv message will be stored
  comm_buffer_t m_storage;

  //! data size per quad (i.e. number of cells per leaf times number of variables)
  //! this value is set each time we call allocate_send_recv_buffers
  size_t m_data_size_per_quad;

}; // class MeshGhostsExchanger

// explicit template instantiation
extern template class MeshGhostsExchanger<2, real_t, DefaultDevice>;
extern template class MeshGhostsExchanger<3, real_t, DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_MESHGHOSTSEXCHANGER_H_
