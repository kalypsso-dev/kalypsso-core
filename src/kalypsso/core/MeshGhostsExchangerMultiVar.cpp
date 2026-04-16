// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshGhostsExchangerMultiVar.cpp
 */

#include <kalypsso/core/MeshGhostsExchangerMultiVar.h>

#include <kalypsso/core/DataArrayUtils.h>

namespace kalypsso
{
// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::resize_mv(int32_t num_block_mirrors,
                                                         int32_t num_block_ghosts,
                                                         size_t  data_size_per_block)
{
  const auto current_capacity = this->m_storage.size();
  const auto num_blocks =
    static_cast<size_t>(num_block_mirrors) + static_cast<size_t>(num_block_ghosts);
  const auto requested_capacity = data_size_per_block * num_blocks;

  if (requested_capacity > current_capacity)
  {
    // enlarge capacity by factor capacity_growth_rate
    size_t new_storage_capacity =
      DataArrayUtils::allocated_capacity(static_cast<size_t>(requested_capacity));

    Kokkos::resize(
      Kokkos::view_alloc(Kokkos::WithoutInitializing), this->m_storage, new_storage_capacity);
  }

  const auto offset = data_size_per_block * static_cast<size_t>(num_block_mirrors);
  // clang-format off
  this->m_send =
    comm_buffer_unmanaged_t(this->m_storage.data(),
                            data_size_per_block * static_cast<size_t>(num_block_mirrors));
  this->m_recv =
    comm_buffer_unmanaged_t(this->m_storage.data() + offset,
                            data_size_per_block * static_cast<size_t>(num_block_ghosts));
  // clang-format on

  // This is essentially what the value is
  m_data_size_per_block = data_size_per_block;

} // resize_mv

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::resize_offsets()
{
  const auto offsets_capacity_current = m_offsets_storage.size();

  const auto offsets_capacity_required =
    (static_cast<size_t>(this->m_amr_mesh.local_num_ghosts()) + 1) +
    (static_cast<size_t>(this->m_amr_mesh.local_num_mirrors()) + 1);

  if (offsets_capacity_required > offsets_capacity_current)
  {
    auto offsets_capacity_new =
      DataArrayUtils::allocated_capacity(static_cast<size_t>(offsets_capacity_required));

    Kokkos::resize(
      Kokkos::view_alloc(Kokkos::WithoutInitializing), m_offsets_storage, offsets_capacity_new);
    Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                   m_offsets_storage_host,
                   offsets_capacity_new);
  }

  m_ghost_offsets = offset_dev_unmanaged_t(
    m_offsets_storage.data(), static_cast<size_t>(this->m_amr_mesh.local_num_ghosts()) + 1);

  m_mirror_offsets = offset_dev_unmanaged_t(
    m_offsets_storage.data() + static_cast<size_t>(this->m_amr_mesh.local_num_ghosts()) + 1,
    static_cast<size_t>(this->m_amr_mesh.local_num_mirrors()) + 1);

  m_ghost_offsets_host = offset_host_unmanaged_t(
    m_offsets_storage_host.data(), static_cast<size_t>(this->m_amr_mesh.local_num_ghosts()) + 1);

  m_mirror_offsets_host = offset_host_unmanaged_t(
    m_offsets_storage_host.data() + static_cast<size_t>(this->m_amr_mesh.local_num_ghosts()) + 1,
    static_cast<size_t>(this->m_amr_mesh.local_num_mirrors()) + 1);

} // resize_offsets

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
auto
MeshGhostsExchangerMultiVar<dim, T, device_t>::allocated_size_in_bytes() const -> size_t
{
  auto size = MeshGhostsExchanger<dim, T, device_t>::allocated_size_in_bytes();

  size += m_offsets_storage.size() * sizeof(typename offset_dev_t::non_const_value_type);

  return size;
}

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::pack_mirror_data_multi_var(
  DataArrayBlockMultiVar_t           userdata_block,
  int32_t                            num_block_mirrors,
  orchard_key_view_t                 mirror_keys_device,
  Kokkos::View<uint32_t *, device_t> mirror_offsets)
{
  auto amr_hashmap = this->m_mesh_map.hashmap();
  auto num_mirror_quad = this->m_amr_mesh.local_num_mirrors();

  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();

  [[maybe_unused]] auto local_num_quadrants = this->m_amr_mesh.local_num_quadrants();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(num_cells * num_block_mirrors) == this->m_send.extent(0),
          "[MeshGhostsExchangerMultiVar::pack_mirror_data_multi_var] send buffer has wrong size");

  auto mir_data = DataArrayBlockUnmanaged_t(
    this->m_send.data(), bSizes, 1, static_cast<size_t>(num_block_mirrors));

  {
    // traverse the list of mirror keys, and for each key try to find it in the old hashmap
    Kokkos::parallel_for(
      "pack_mirror_data_multi_var - block - cell center data",
      Kokkos::RangePolicy<exec_space>(0, num_mirror_quad * num_cells),
      KOKKOS_LAMBDA(const int32_t & global_index) {
        const auto imirror = global_index / num_cells;
        const auto icell = global_index - imirror * num_cells;
        auto       key = mirror_keys_device(imirror);

        // find key index in map
        auto key_index_map = amr_hashmap.find(key);

        // first check if key exists in the hashmap, if it exists, it means we found a matching
        // owned mirror quadrant
        if (amr_hashmap.valid_at(key_index_map))
        {
          auto userdata_index = static_cast<int32_t>(amr_hashmap.value_at(key_index_map));
          KOKKOS_ASSERT(
            userdata_index < local_num_quadrants &&
            "[MeshGhostsExchangerMultiVar::pack_mirror_data_multi_var] index must correspond "
            "to a locally owned quadrant.");
          const auto nb_var = userdata_block.num_vars(userdata_index);
          const auto start_block = static_cast<int32_t>(mirror_offsets(imirror));
          for (int32_t ivar = 0; ivar < nb_var; ++ivar)
            mir_data(icell, 0, start_block + ivar) = userdata_block(icell, ivar, userdata_index);
        }
      });
  }

  Kokkos::fence();

} // pack_mirror_data_multi_var - block (cell-center) version - new

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::unpack_ghost_data_multi_var(
  DataArrayBlockMultiVar_t userdata_block,
  int32_t                  num_block_owned,
  int32_t                  num_block_ghosts)
{
  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();

  // make sure m_recv buffer was allocated with the right size
  assertm(
    static_cast<size_t>(num_block_ghosts * num_cells) == this->m_recv.extent(0),
    "[MeshGhostsExchangerMultiVar::unpack_ghost_data_multi_var] receive buffer has wrong size");

  auto recv_data = DataArrayBlockUnmanaged_t(
    this->m_recv.data(), bSizes, 1, static_cast<size_t>(num_block_ghosts));

  auto recv_range = std::pair<std::size_t, std::size_t>(
    num_cells * num_block_owned, num_cells * (num_block_owned + num_block_ghosts));
  auto userdata_block_ghost = Kokkos::subview(userdata_block.storage().logical_view(), recv_range);

  Kokkos::deep_copy(userdata_block_ghost, recv_data.logical_view());

  Kokkos::fence();
} // unpack_ghost_data_multi_var - cell-center block version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::do_mpi_send_recv_multi_var(
  const Kokkos::View<uint32_t *, HostDevice> & ghost_offsets,
  const Kokkos::View<uint32_t *, HostDevice> & mirror_offsets)
{
  // total number of MPI procs
  auto num_procs = this->m_par_env.size();

  // vector of MPI request for asynchronous send and receives
  std::vector<MPI_Request> requests;

  uint32_t num_requests_recv = 0;
  uint32_t num_requests_send = 0;

  // the following could be factorized in AMRmesh class (where the ghost object is stored)
  //
  // Compute the total number of MPI request (send and receive)
  //
  for (auto iproc = 0; iproc < num_procs; ++iproc)
  {
    auto nghosts = this->m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);
    if (nghosts > 0)
    {
      num_requests_recv++;
    }

    auto nmirrors = this->m_amr_mesh.local_num_mirrors(iproc);
    KALYPSSO_ASSERT(nmirrors >= 0);
    if (nmirrors > 0)
    {
      num_requests_send++;
    }
  }
  requests.reserve(num_requests_recv + num_requests_send);

  //
  // Start to receive ghost data. num_recv is the total number of message to receive.
  //
  uint32_t num_recv = 0;
  for (auto iproc = 0; iproc < num_procs; ++iproc)
  {
    // number of ghosts quadrants owned by remote MPI process iproc
    auto nghosts = this->m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);

    // if nghosts = 0, it means MPI process don't actually own ghosts, we won't receive any data
    // from MPI process iproc

    // remote MPI process iproc actually owns interesting data, we want to receive those data
    if (nghosts > 0)
    {
      const auto offset = this->m_amr_mesh.ghost()->proc_offsets[iproc];
      auto       recv_range = std::pair<std::size_t, std::size_t>(
        ghost_offsets(offset) * m_data_size_per_block,
        ghost_offsets(offset + nghosts) * m_data_size_per_block);
      auto recv_buff = Kokkos::subview(this->m_recv, recv_range);

      requests[num_recv] =
        this->m_par_env.comm().MPI_Irecv(recv_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

      num_recv++;
    }
  } // end for iproc

  //
  // Start to send mirror data. num_send is the total number of message to send.
  //
  uint32_t num_send = 0;
  for (auto iproc = 0; iproc < num_procs; ++iproc)
  {
    // number of mirror quadrants owned by current MPI process to be filled
    auto nmirrors = this->m_amr_mesh.local_num_mirrors(iproc);
    KALYPSSO_ASSERT(nmirrors >= 0);

    // if nmirrors = 0, it means current MPI process don't need to send data to MPI process
    // iproc

    if (nmirrors > 0)
    {
      const auto offset = this->m_amr_mesh.ghost()->mirror_proc_offsets[iproc];
      auto       send_range = std::pair<std::size_t, std::size_t>(
        mirror_offsets(offset) * m_data_size_per_block,
        mirror_offsets(offset + nmirrors) * m_data_size_per_block);
      auto send_buff = Kokkos::subview(this->m_send, send_range);

      requests[num_requests_recv + num_send] =
        this->m_par_env.comm().MPI_Isend(send_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

      num_send++;
    }
  } // end for iproc

  //
  // let's wait for all send/recv comm to finish
  //
  this->m_par_env.comm().MPI_Waitall(num_requests_recv + num_requests_send, requests.data());

} // do_mpi_send_recv_multi_var

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchangerMultiVar<dim, T, device_t>::exchange_multi_var(
  DataArrayBlockMultiVar_t userdata_block)
{
  //! TODO: Would it be better to not copy the offsets to the host?

  auto mirror_keys = this->m_mesh_map.mirror_orchard_keys();
  auto hashmap = this->m_mesh_map.hashmap();

  resize_offsets();

  auto ghost_offsets = m_ghost_offsets;
  auto mirror_offsets = m_mirror_offsets;

  const auto start_ghost = this->m_amr_mesh.local_num_quadrants();
  Kokkos::parallel_for(
    "init ghost offsets",
    Kokkos::RangePolicy<exec_space>(0, this->m_amr_mesh.local_num_ghosts() + 1),
    KOKKOS_LAMBDA(const int32_t i_oct) {
      auto start_offset = userdata_block.offsets()(start_ghost);
      ghost_offsets(i_oct) = userdata_block.offsets()(start_ghost + i_oct) - start_offset;
    });

  const auto num_mirrors = this->m_amr_mesh.local_num_mirrors();
  Kokkos::parallel_scan(
    "init mirror offsets",
    Kokkos::RangePolicy<exec_space>(0, num_mirrors + 1),
    KOKKOS_LAMBDA(const int32_t i_oct, uint32_t & offset, bool is_final) {
      if (is_final)
        mirror_offsets(i_oct) = offset;
      if (i_oct != num_mirrors)
      {
        const auto key = mirror_keys(i_oct);
        const auto orig_oct = hashmap.value_at(hashmap.find(key));
        offset += static_cast<uint32_t>(userdata_block.num_vars(orig_oct));
      }
    });

  Kokkos::deep_copy(m_ghost_offsets_host, m_ghost_offsets);
  Kokkos::deep_copy(m_mirror_offsets_host, m_mirror_offsets);

  const int32_t num_block_ghosts =
    static_cast<int32_t>(m_ghost_offsets_host(m_ghost_offsets_host.size() - 1));

  const int32_t num_block_mirrors =
    static_cast<int32_t>(m_mirror_offsets_host(m_mirror_offsets_host.size() - 1));

  uint32_t num_block_owned = 0;
  Kokkos::deep_copy(num_block_owned,
                    Kokkos::subview(userdata_block.offsets().logical_view(), start_ghost));

  // check if (re-)allocation needed
  const auto data_size_per_block = static_cast<uint32_t>(userdata_block.num_cells());

  resize_mv(num_block_mirrors, num_block_ghosts, data_size_per_block);

  pack_mirror_data_multi_var(userdata_block, num_block_mirrors, mirror_keys, m_mirror_offsets);

  do_mpi_send_recv_multi_var(m_ghost_offsets_host, m_mirror_offsets_host);

  unpack_ghost_data_multi_var(
    userdata_block, static_cast<int32_t>(num_block_owned), num_block_ghosts);

} // exchange_multi_var

// explicit template instantiation
template class MeshGhostsExchangerMultiVar<2, real_t, DefaultDevice>;
template class MeshGhostsExchangerMultiVar<3, real_t, DefaultDevice>;

} // namespace kalypsso
