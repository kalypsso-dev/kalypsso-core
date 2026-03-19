// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MeshGhostsExchanger.cpp
 */

#include <kalypsso/core/MeshGhostsExchanger.h>

#include <kalypsso/core/DataArrayUtils.h>

namespace kalypsso
{
// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::resize(size_t data_size_per_quad)
{
  const auto current_capacity = m_storage.size();
  const auto num_blocks = static_cast<size_t>(m_amr_mesh.local_num_mirrors()) +
                          static_cast<size_t>(m_amr_mesh.local_num_ghosts());
  const auto requested_capacity = data_size_per_quad * num_blocks;

  if (requested_capacity > current_capacity)
  {
    // enlarge capacity by factor capacity_growth_rate
    size_t new_storage_capacity =
      DataArrayUtils::allocated_capacity(static_cast<size_t>(requested_capacity));

    Kokkos::resize(
      Kokkos::view_alloc(Kokkos::WithoutInitializing), m_storage, new_storage_capacity);
  }

  const auto offset = data_size_per_quad * static_cast<size_t>(m_amr_mesh.local_num_mirrors());
  // clang-format off
  m_send =
    comm_buffer_unmanaged_t(m_storage.data(),
                            data_size_per_quad * static_cast<size_t>(m_amr_mesh.local_num_mirrors()));
  m_recv =
    comm_buffer_unmanaged_t(m_storage.data() + offset,
                            data_size_per_quad * static_cast<size_t>(m_amr_mesh.local_num_ghosts()));
  // clang-format on

  // This is essentially what the value is
  m_data_size_per_quad = data_size_per_quad;

} // resize

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
auto
MeshGhostsExchanger<dim, T, device_t>::allocated_size_in_bytes() const -> size_t
{
  size_t size = m_storage.size() * sizeof(T);
  return size;
}

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::pack_mirror_data(DataArrayLeaf_t    userdata_leaf,
                                                        orchard_key_view_t mirror_keys_device)
{

  auto local_num_mirrors = m_amr_mesh.local_num_mirrors();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(local_num_mirrors) * userdata_leaf.extent(1) == m_send.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] send buffer has wrong size");

  [[maybe_unused]] auto local_num_quadrants = m_amr_mesh.local_num_quadrants();

  // number of scalar values per quad
  auto num_vars = userdata_leaf.extent(1);

  DataArrayLeafUnmanaged_t mir_data = DataArrayLeafUnmanaged_t(
    m_send.data(), static_cast<size_t>(local_num_mirrors), static_cast<size_t>(num_vars));

  auto amr_hashmap = m_mesh_map.hashmap();

  // traverse the list of mirror keys, and for each key try to find it in the old hashmap
  Kokkos::parallel_for(
    "pack_mirror_data - leaf",
    Kokkos::RangePolicy<exec_space>(0, local_num_mirrors),
    KOKKOS_LAMBDA(const int imirror) {
      auto key = mirror_keys_device(imirror);

      // find key index in map
      auto key_index_map = amr_hashmap.find(key);

      // first check if key exists in the hashmap, if it exists, it means we found a matching
      // owned mirror quadrant
      if (amr_hashmap.valid_at(key_index_map))
      {
        auto index = amr_hashmap.value_at(key_index_map);

        KOKKOS_ASSERT(index < local_num_quadrants &&
                      "[MeshGhostsExchanger::pack_mirror_data] index must correspond to a locally "
                      "owned quadrant.");

        for (uint32_t ivar = 0; ivar < num_vars; ++ivar)
        {
          mir_data(imirror, ivar) = userdata_leaf(index, ivar);
        }
      }
      else
      {
        // printf("[MeshGhostsExchanger::pack_mirror_data] Invalid key !\n");
      }
    });

  Kokkos::fence();

  // now m_send (aka mir_data) is ready to be sent

} // pack_mirror_data - leaf version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::pack_mirror_data(DataArrayBlock_t   userdata_block,
                                                        orchard_key_view_t mirror_keys_device)
{

  auto amr_hashmap = m_mesh_map.hashmap();
  auto num_mirror_quad = m_amr_mesh.local_num_mirrors();

  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();
  const auto num_vars = userdata_block.num_vars();

  [[maybe_unused]] auto local_num_quadrants = m_amr_mesh.local_num_quadrants();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(num_cells * num_vars * num_mirror_quad) == m_send.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] send buffer has wrong size");

  auto mir_data = DataArrayBlockUnmanaged_t(m_send.data(), bSizes, num_vars, num_mirror_quad);

  {
    // traverse the list of mirror keys, and for each key try to find it in the old hashmap
    Kokkos::parallel_for(
      "pack_mirror_data - block - cell center data",
      Kokkos::RangePolicy<exec_space>(0, num_mirror_quad * num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        const auto imirror = global_index / num_cells;
        const auto icell = global_index - imirror * num_cells;
        auto       key = mirror_keys_device(imirror);

        // find key index in map
        auto key_index_map = amr_hashmap.find(key);

        // first check if key exists in the hashmap, if it exists, it means we found a matching
        // owned mirror quadrant
        if (amr_hashmap.valid_at(key_index_map))
        {
          auto userdata_index = amr_hashmap.value_at(key_index_map);
          KOKKOS_ASSERT(
            userdata_index < local_num_quadrants &&
            "[MeshGhostsExchanger::pack_mirror_data] index must correspond to a locally "
            "owned quadrant.");
          for (int32_t ivar = 0; ivar < num_vars; ++ivar)
          {
            mir_data(icell, ivar, imirror) = userdata_block(icell, ivar, userdata_index);
          }
        }
      });
  }
  Kokkos::fence();

  // now m_send (aka mir_data) is ready to be sent

} // pack_mirror_data - block (cell-center) version - new

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::pack_mirror_data_multi_var(
  DataArrayBlockMultiVar_t           userdata_block,
  int32_t                            num_block_mirrors,
  orchard_key_view_t                 mirror_keys_device,
  Kokkos::View<uint32_t *, device_t> mirror_offsets)
{
  auto amr_hashmap = m_mesh_map.hashmap();
  auto num_mirror_quad = m_amr_mesh.local_num_mirrors();

  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();

  [[maybe_unused]] auto local_num_quadrants = m_amr_mesh.local_num_quadrants();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(num_cells * num_block_mirrors) == m_send.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data_multi_var] send buffer has wrong size");

  auto mir_data =
    DataArrayBlockUnmanaged_t(m_send.data(), bSizes, 1, static_cast<size_t>(num_block_mirrors));

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
          KOKKOS_ASSERT(userdata_index < local_num_quadrants &&
                        "[MeshGhostsExchanger::pack_mirror_data_multi_var] index must correspond "
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
MeshGhostsExchanger<dim, T, device_t>::pack_mirror_data(FaceDataArrayBlock_t face_userdata,
                                                        orchard_key_view_t   mirror_keys_device)
{

  auto amr_hashmap = m_mesh_map.hashmap();
  auto num_mirror_quad = m_amr_mesh.local_num_mirrors();

  // number of face elements per block (total number of faces)
  auto num_elts_per_octant = face_userdata.num_elements_per_octant();

  [[maybe_unused]] auto local_num_quadrants = m_amr_mesh.local_num_quadrants();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(num_elts_per_octant * num_mirror_quad) == m_send.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] send buffer has wrong size");

  // create an unmanaged 1D view of data to send
  FaceFlatArrayUnmanaged_t mir_data = FaceFlatArrayUnmanaged_t(
    m_send.data(), static_cast<size_t>(num_elts_per_octant * num_mirror_quad));

  // get 1D view of face data
  auto userdata = face_userdata.logical_view();

  {
    // traverse the list of mirror keys, and for each key try to find it in the old hashmap
    Kokkos::parallel_for(
      "pack_mirror_data - block - face center data",
      Kokkos::RangePolicy<exec_space>(0, num_mirror_quad * num_elts_per_octant),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        const auto imirror = global_index / num_elts_per_octant;
        const auto ielt = global_index - imirror * num_elts_per_octant;
        auto       key = mirror_keys_device(imirror);

        // find key index in map
        auto key_index_map = amr_hashmap.find(key);

        // first check if key exists in the hashmap, if it exists, it means we found a matching
        // owned mirror quadrant
        if (amr_hashmap.valid_at(key_index_map))
        {
          auto userdata_index = amr_hashmap.value_at(key_index_map);
          KOKKOS_ASSERT(
            userdata_index < local_num_quadrants &&
            "[MeshGhostsExchanger::pack_mirror_data] index must correspond to a locally "
            "owned quadrant.");
          {
            mir_data(ielt + num_elts_per_octant * imirror) =
              userdata(ielt + num_elts_per_octant * userdata_index);
          }
        }
      });
  }
  Kokkos::fence();

  // now m_send (aka mir_data) is ready to be sent

} // pack_mirror_data - block (face-center) version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::unpack_ghost_data(DataArrayLeaf_t userdata_leaf)
{
  // get the number of owned quadrants in local MPI process
  auto num_owned_quad = m_amr_mesh.local_num_quadrants();

  // get the number of ghost quadrants in local MPI process
  auto num_ghost_quad = m_amr_mesh.local_num_ghosts();

  // number of scalar values per quad
  [[maybe_unused]] auto num_vars = userdata_leaf.extent(1);

  // make sure m_recv buffer was allocated with the right size
  assertm(static_cast<size_t>(num_ghost_quad) * num_vars == m_recv.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] send buffer has wrong size");

  DataArrayLeafUnmanaged_t recv_data = DataArrayLeafUnmanaged_t(
    m_recv.data(), static_cast<size_t>(num_ghost_quad), static_cast<size_t>(num_vars));

  auto recv_range =
    std::pair<std::size_t, std::size_t>(num_owned_quad, num_owned_quad + num_ghost_quad);
  auto userdata_leaf_ghost = Kokkos::subview(userdata_leaf, recv_range, Kokkos::ALL);

  Kokkos::deep_copy(userdata_leaf_ghost, recv_data);

  Kokkos::fence();

} // unpack_ghost_data - leaf version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::unpack_ghost_data(DataArrayBlock_t userdata_block)
{
  // get the number of owned quadrants in local MPI process
  const auto num_owned_quad = m_amr_mesh.local_num_quadrants();

  // get the number of ghost quadrants in local MPI process
  const auto num_ghost_quad = m_amr_mesh.local_num_ghosts();

  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();
  const auto num_vars = userdata_block.num_vars();
  const auto num_elts_per_quad = num_cells * num_vars;

  // make sure m_recv buffer was allocated with the right size
  assertm(static_cast<size_t>(num_ghost_quad * num_vars * num_cells) == m_recv.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] receive buffer has wrong size");

  auto recv_data = DataArrayBlockUnmanaged_t(m_recv.data(), bSizes, num_vars, num_ghost_quad);

  auto recv_range = std::pair<std::size_t, std::size_t>(
    num_elts_per_quad * num_owned_quad, num_elts_per_quad * (num_owned_quad + num_ghost_quad));
  auto userdata_block_ghost = Kokkos::subview(userdata_block.logical_view(), recv_range);

  Kokkos::deep_copy(userdata_block_ghost, recv_data.logical_view());

  Kokkos::fence();

} // unpack_ghost_data - cell-center block version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::unpack_ghost_data_multi_var(
  DataArrayBlockMultiVar_t userdata_block,
  int32_t                  num_block_owned,
  int32_t                  num_block_ghosts)
{
  // number of cells per block, and scalar values per cell
  const auto num_cells = userdata_block.num_cells();
  const auto bSizes = userdata_block.block_size();

  // make sure m_recv buffer was allocated with the right size
  assertm(static_cast<size_t>(num_block_ghosts * num_cells) == m_recv.extent(0),
          "[MeshGhostsExchanger::unpack_ghost_data_multi_var] receive buffer has wrong size");

  auto recv_data =
    DataArrayBlockUnmanaged_t(m_recv.data(), bSizes, 1, static_cast<size_t>(num_block_ghosts));

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
MeshGhostsExchanger<dim, T, device_t>::unpack_ghost_data(FaceDataArrayBlock_t face_userdata)
{
  // get the number of owned quadrants in local MPI process
  const auto num_owned_quad = m_amr_mesh.local_num_quadrants();

  // get the number of ghost quadrants in local MPI process
  const auto num_ghost_quad = m_amr_mesh.local_num_ghosts();

  // number of face elements per block (total number of faces)
  const auto num_elts_per_octant = face_userdata.num_elements_per_octant();

  // make sure m_recv buffer was allocated with the right size
  assertm(static_cast<size_t>(num_elts_per_octant * num_ghost_quad) == m_recv.extent(0),
          "[MeshGhostsExchanger::pack_mirror_data] receive buffer has wrong size");

  FaceFlatArrayUnmanaged_t recv_data = FaceFlatArrayUnmanaged_t(
    m_recv.data(), static_cast<size_t>(num_elts_per_octant * num_ghost_quad));

  // get 1D view of face data
  auto userdata = face_userdata.logical_view();

  auto recv_range = std::pair<std::size_t, std::size_t>(
    num_elts_per_octant * num_owned_quad, num_elts_per_octant * (num_owned_quad + num_ghost_quad));
  auto userdata_ghost = Kokkos::subview(userdata, recv_range);

  Kokkos::deep_copy(userdata_ghost, recv_data);

  Kokkos::fence();

} // unpack_ghost_data - face-center block version

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::do_mpi_send_recv()
{
  // total number of MPI procs
  auto num_procs = m_par_env.size();

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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);
    if (nghosts > 0)
    {
      num_requests_recv++;
    }

    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);

    // if nghosts = 0, it means MPI process don't actually own ghosts, we won't receive any data
    // from MPI process iproc

    // remote MPI process iproc actually owns interesting data, we want to receive those data
    if (nghosts > 0)
    {
      auto recv_range = std::pair<std::size_t, std::size_t>(
        static_cast<size_t>(m_amr_mesh.ghost()->proc_offsets[iproc]) * m_data_size_per_quad,
        static_cast<size_t>((m_amr_mesh.ghost()->proc_offsets[iproc] + nghosts)) *
          m_data_size_per_quad);
      auto recv_buff = Kokkos::subview(m_recv, recv_range);

      requests[num_recv] =
        m_par_env.comm().MPI_Irecv(recv_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

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
    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
    KALYPSSO_ASSERT(nmirrors >= 0);

    // if nmirrors = 0, it means current MPI process don't need to send data to MPI process
    // iproc

    if (nmirrors > 0)
    {
      auto send_range = std::pair<std::size_t, std::size_t>(
        static_cast<size_t>(m_amr_mesh.ghost()->mirror_proc_offsets[iproc]) * m_data_size_per_quad,
        static_cast<size_t>((m_amr_mesh.ghost()->mirror_proc_offsets[iproc] + nmirrors)) *
          m_data_size_per_quad);
      auto send_buff = Kokkos::subview(m_send, send_range);

      requests[num_requests_recv + num_send] =
        m_par_env.comm().MPI_Isend(send_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

      num_send++;
    }
  } // end for iproc

  //
  // let's wait for all send/recv comm to finish
  //
  m_par_env.comm().MPI_Waitall(num_requests_recv + num_requests_send, requests.data());

} // do_mpi_send_recv

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::do_mpi_send_recv_multi_var(
  const Kokkos::View<uint32_t *, HostDevice> & ghost_offsets,
  const Kokkos::View<uint32_t *, HostDevice> & mirror_offsets)
{
  // total number of MPI procs
  auto num_procs = m_par_env.size();

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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);
    if (nghosts > 0)
    {
      num_requests_recv++;
    }

    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);

    // if nghosts = 0, it means MPI process don't actually own ghosts, we won't receive any data
    // from MPI process iproc

    // remote MPI process iproc actually owns interesting data, we want to receive those data
    if (nghosts > 0)
    {
      const auto offset = m_amr_mesh.ghost()->proc_offsets[iproc];
      auto       recv_range =
        std::pair<std::size_t, std::size_t>(ghost_offsets(offset) * m_data_size_per_quad,
                                            ghost_offsets(offset + nghosts) * m_data_size_per_quad);
      auto recv_buff = Kokkos::subview(m_recv, recv_range);

      requests[num_recv] =
        m_par_env.comm().MPI_Irecv(recv_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

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
    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
    KALYPSSO_ASSERT(nmirrors >= 0);

    // if nmirrors = 0, it means current MPI process don't need to send data to MPI process
    // iproc

    if (nmirrors > 0)
    {
      const auto offset = m_amr_mesh.ghost()->mirror_proc_offsets[iproc];
      auto       send_range = std::pair<std::size_t, std::size_t>(
        mirror_offsets(offset) * m_data_size_per_quad,
        mirror_offsets(offset + nmirrors) * m_data_size_per_quad);
      auto send_buff = Kokkos::subview(m_send, send_range);

      requests[num_requests_recv + num_send] =
        m_par_env.comm().MPI_Isend(send_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

      num_send++;
    }
  } // end for iproc

  //
  // let's wait for all send/recv comm to finish
  //
  m_par_env.comm().MPI_Waitall(num_requests_recv + num_requests_send, requests.data());

} // do_mpi_send_recv_multi_var

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::do_mpi_send_recv_inplace(
  DataArrayGhostedBlock_t userdata_ghosted_block)
{
  const auto data_size_per_quad =
    userdata_ghosted_block.num_cells() * userdata_ghosted_block.num_vars();

  auto userdata_view = userdata_ghosted_block.flat_view();

  // total number of MPI procs
  const auto num_procs = m_par_env.size();

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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);
    if (nghosts > 0)
    {
      num_requests_recv++;
    }

    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
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
    auto nghosts = m_amr_mesh.local_num_ghosts(iproc);
    KALYPSSO_ASSERT(nghosts >= 0);

    // if nghosts = 0, it means MPI process don't actually own ghosts, we won't receive any data
    // from MPI process iproc

    // MPI process iproc actually owns interesting data, we want to receive those data
    if (nghosts > 0)
    {
      auto recv_offset =
        (m_amr_mesh.local_num_mirrors() + m_amr_mesh.ghost()->proc_offsets[iproc]) *
        data_size_per_quad;
      auto recv_size = nghosts * data_size_per_quad;

      // declare recv buffer as unmanaged kokkos view used for inplace transfer
      comm_buffer_unmanaged_t recv_buff(userdata_view.data() + recv_offset,
                                        static_cast<size_t>(recv_size));

      requests[num_recv] =
        m_par_env.comm().MPI_Irecv(recv_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

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
    auto nmirrors = m_amr_mesh.local_num_mirrors(iproc);
    KALYPSSO_ASSERT(nmirrors >= 0);

    // if nmirrors = 0, it means current MPI process don't need to send data to MPI process
    // iproc

    if (nmirrors > 0)
    {
      auto send_offset = (m_amr_mesh.ghost()->mirror_proc_offsets[iproc]) * data_size_per_quad;
      auto send_size = nmirrors * data_size_per_quad;

      // declare send buffer as unmanaged kokkos view used for inplace transfer
      comm_buffer_unmanaged_t send_buff(userdata_view.data() + send_offset,
                                        static_cast<size_t>(send_size));

      requests[num_requests_recv + num_send] =
        m_par_env.comm().MPI_Isend(send_buff, iproc, KALYPSSO_COMM_GHOST_EXCHANGE_TAG);

      num_send++;
    }
  } // end for iproc

  //
  // let's wait for all send/recv comm to finish
  //
  m_par_env.comm().MPI_Waitall(num_requests_recv + num_requests_send, requests.data());

} // do_mpi_send_recv_inplace

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::exchange(DataArrayLeaf_t userdata_leaf)
{

  resize(userdata_leaf.extent(1));

  pack_mirror_data(userdata_leaf, m_mesh_map.mirror_orchard_keys());

  do_mpi_send_recv();

  unpack_ghost_data(userdata_leaf);

} // exchange

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::exchange(DataArrayBlock_t userdata_block)
{

  // check if (re-)allocation needed
  const auto data_size_per_quad =
    static_cast<uint32_t>(userdata_block.num_cells() * userdata_block.num_vars());

  resize(data_size_per_quad);

  pack_mirror_data(userdata_block, m_mesh_map.mirror_orchard_keys());

  do_mpi_send_recv();

  unpack_ghost_data(userdata_block);

} // exchange

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::exchange(FaceDataArrayBlock_t userdata_block_face)
{

  // check if (re-)allocation needed
  const auto data_size_per_quad = userdata_block_face.num_elements_per_octant();

  resize(static_cast<size_t>(data_size_per_quad));

  pack_mirror_data(userdata_block_face, m_mesh_map.mirror_orchard_keys());

  do_mpi_send_recv();

  unpack_ghost_data(userdata_block_face);

} // exchange

// =========================================================================
// =========================================================================
template <size_t dim, typename T, typename device_t>
void
MeshGhostsExchanger<dim, T, device_t>::exchange_multi_var(DataArrayBlockMultiVar_t userdata_block)
{
  //! TODO: Would it be better to not copy the offsets to the host?
  using HostSpace = typename HostDevice::execution_space;

  auto mirror_keys = m_mesh_map.mirror_orchard_keys();
  auto hashmap = m_mesh_map.hashmap();

  Kokkos::View<uint32_t *, device_t> ghost_offsets(
    "ghost offsets", static_cast<size_t>(m_amr_mesh.local_num_ghosts()) + 1);

  Kokkos::View<uint32_t *, device_t> mirror_offsets(
    "mirror offsets", static_cast<size_t>(m_amr_mesh.local_num_mirrors()) + 1);

  const auto start_ghost = m_amr_mesh.local_num_quadrants();
  Kokkos::parallel_for(
    "init ghost offsets",
    Kokkos::RangePolicy<exec_space>(0, m_amr_mesh.local_num_ghosts() + 1),
    KOKKOS_LAMBDA(const int32_t i_oct) {
      auto start_offset = userdata_block.offsets()(start_ghost);
      ghost_offsets(i_oct) = userdata_block.offsets()(start_ghost + i_oct) - start_offset;
    });

  const auto num_mirrors = m_amr_mesh.local_num_mirrors();
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

  auto ghost_offsets_host = Kokkos::create_mirror_view_and_copy(HostSpace{}, ghost_offsets);
  auto mirror_offsets_host = Kokkos::create_mirror_view_and_copy(HostSpace{}, mirror_offsets);

  const int32_t num_block_ghosts =
    static_cast<int32_t>(ghost_offsets_host(ghost_offsets.size() - 1));
  const int32_t num_block_mirrors =
    static_cast<int32_t>(mirror_offsets_host(mirror_offsets.size() - 1));
  uint32_t num_block_owned = 0;
  Kokkos::deep_copy(num_block_owned,
                    Kokkos::subview(userdata_block.offsets().logical_view(), start_ghost));

  // check if (re-)allocation needed
  const auto data_size_per_block = static_cast<uint32_t>(userdata_block.num_cells());

  resize(data_size_per_block);

  pack_mirror_data_multi_var(userdata_block, num_block_mirrors, mirror_keys, mirror_offsets);

  do_mpi_send_recv_multi_var(ghost_offsets_host, mirror_offsets_host);

  unpack_ghost_data_multi_var(
    userdata_block, static_cast<int32_t>(num_block_owned), num_block_ghosts);

} // exchange_multi_var

// explicit template instantiation
template class MeshGhostsExchanger<2, real_t, DefaultDevice>;
template class MeshGhostsExchanger<3, real_t, DefaultDevice>;

} // namespace kalypsso
