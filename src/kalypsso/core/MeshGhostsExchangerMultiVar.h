// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshGhostsExchangerMultiVar.h
 */
#ifndef KALYPSSO_CORE_MESHGHOSTSEXCHANGERMULTIVAR_H_
#define KALYPSSO_CORE_MESHGHOSTSEXCHANGERMULTIVAR_H_

#include <kalypsso/core/MeshGhostsExchanger.h>

namespace kalypsso
{

/**
 * \class MeshGhostsExchangerMultiVar
 *
 * \copydoc MeshGhostsExchanger
 *
 * Main differences with MeshGhostExchanger are
 * - using "data-per-block" instead of "data-per-octant".
 * - additional array for storing offsets.
 *
 */
template <size_t dim, typename T, typename device_t>
class MeshGhostsExchangerMultiVar : public MeshGhostsExchanger<dim, T, device_t>
{

public:
  using exec_space = typename device_t::execution_space;

  //! type alias to access p4est C API (2D or 3D)
  using p4est_t = typename p4est::Wrapper<dim>;

  using forest_t = typename p4est_t::forest_t;
  using ghost_t = typename p4est_t::ghost_t;

  using DataArrayBlock_t = DataArrayBlock<dim, T, device_t>;
  using DataArrayBlockUnmanaged_t = DataArrayBlock<dim, T, device_t, StorageType::UNMANAGED>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, T, device_t>;

  using MaterialPresenceView_t = MaterialPresenceView<device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, T, device_t>;

  using comm_buffer_t = Kokkos::View<T *, device_t>;
  using comm_buffer_unmanaged_t =
    Kokkos::View<T *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

  using offset_dev_t = Kokkos::View<uint32_t *, device_t>;
  using offset_dev_unmanaged_t =
    Kokkos::View<uint32_t *, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using offset_host_t = Kokkos::View<uint32_t *, HostDevice>;
  using offset_host_unmanaged_t =
    Kokkos::View<uint32_t *, HostDevice, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


  // =========================================================================
  // =========================================================================
  //! constructor
  MeshGhostsExchangerMultiVar(const ConfigMap &        config_map,
                              const ParallelEnv &      par_env,
                              AMRmesh<dim> &           amr_mesh,
                              MeshMap<dim, device_t> & mesh_map)
    : MeshGhostsExchanger<dim, T, device_t>(config_map, par_env, amr_mesh, mesh_map)
    , m_offsets_storage("offsets (device storage)", 0)
    , m_offsets_storage_host("offsets (host storage)", 0)
    , m_ghost_offsets(nullptr, 0)
    , m_ghost_offsets_host(nullptr, 0)
    , m_mirror_offsets(nullptr, 0)
    , m_mirror_offsets_host(nullptr, 0)
    , m_data_size_per_block(0)

  {}

  // =========================================================================
  // =========================================================================
  //! destructor
  ~MeshGhostsExchangerMultiVar() = default;

  // =========================================================================
  // =========================================================================
  //! resize all buffer for performing ghost data exchange in the multivar case.
  //!
  //! \param[in] num_block_mirrors number of mirror blocks
  //! \param[in] num_block_ghosts number of ghosts blocks
  //! \param[in] data_size_per_block is the number of element per block (i.e. number of cells in an
  //!            octant) to transfer
  void
  resize_mv(int32_t num_block_mirrors, int32_t num_block_ghosts, size_t data_size_per_block);

  // =========================================================================
  // =========================================================================
  //! resize offset buffers.
  void
  resize_offsets();

  // =========================================================================
  // =========================================================================
  /**
   * return amount of allocated memory in bytes
   */
  auto
  allocated_size_in_bytes() const -> size_t override;

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
  //! actually perform MPI comm to send and receive ghost quadrant userdata with multi var
  //!
  //! \param[in] ghost_offsets Host view with the block offsets in m_recv
  //! \param[in] mirror_offsets Host view with the block offsets in m_send
  void
  do_mpi_send_recv_multi_var(const Kokkos::View<uint32_t *, HostDevice> & ghost_offsets,
                             const Kokkos::View<uint32_t *, HostDevice> & mirror_offsets);

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_block is a userdata block array (i.e. one value per cell); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  void
  exchange_multi_var(DataArrayBlockMultiVar_t userdata_block);

private:
  //! device storage for offsets used to locate the begin of each block data in ghost and mirror
  //! octants
  offset_dev_t m_offsets_storage;

  //! host storage for offsets used to locate the begin of each block data in ghost and mirror
  //! octants
  offset_host_t m_offsets_storage_host;

  //! device offsets used to locate the begin of each block data in ghost octants
  offset_dev_unmanaged_t m_ghost_offsets;

  //! host offsets used to locate the begin of each block data in ghost octants
  offset_host_unmanaged_t m_ghost_offsets_host;

  //! device offsets used to locate the begin of each block data in mirror octants
  offset_dev_unmanaged_t m_mirror_offsets;

  //! host offsets used to locate the begin of each block data in mirror octants
  offset_host_unmanaged_t m_mirror_offsets_host;

  //! data size per block (i.e. number of cells per block of cells)
  size_t m_data_size_per_block;

}; // class MeshGhostsExchangerMultiVar

// explicit template instantiation
extern template class MeshGhostsExchangerMultiVar<2, real_t, DefaultDevice>;
extern template class MeshGhostsExchangerMultiVar<3, real_t, DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_MESHGHOSTSEXCHANGERMULTIVAR_H_
