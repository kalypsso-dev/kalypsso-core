// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MaterialPresenceExchanger.cpp
 */

#include <kalypsso/core/MaterialPresenceExchanger.h>

namespace kalypsso
{

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
MaterialPresenceExchanger<dim, device_t>::exchange(
  MaterialPresenceView_t userdata_material_presence)
{

  // check if (re-)allocation needed
  if (static_cast<size_t>(m_amr_mesh.local_num_mirrors()) != m_send.size() or
      static_cast<size_t>(m_amr_mesh.local_num_ghosts()) != m_recv.size())
    allocate_send_recv_buffers();

  pack_mirror_data(userdata_material_presence, m_mesh_map.mirror_orchard_keys());

  do_mpi_send_recv();

  unpack_ghost_data(userdata_material_presence);

} // exchange

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
auto
MaterialPresenceExchanger<dim, device_t>::allocated_size_in_bytes() const -> size_t
{
  return m_send.allocated_size_in_bytes() + m_recv.allocated_size_in_bytes();
}

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
MaterialPresenceExchanger<dim, device_t>::allocate_send_recv_buffers()
{
  // number of mirror quadrants to send in current MPI process
  // IMPORTANT: not to be confused with p4est ghost->mirrors.elem_count
  // which contains all the mirror quadrants without redundancy
  // several mirror quadrants are sent to multiple MPI neighbor process
  // auto num_mirror_quad = m_amr_mesh.ghost()->mirrors.elem_count;
  auto num_mirror_octs = m_amr_mesh.local_num_mirrors();
  m_send.resize(num_mirror_octs);

  // number of ghost quadrants in current MPI process
  auto num_ghost_octs = m_amr_mesh.local_num_ghosts();
  m_recv.resize(num_ghost_octs);

} // allocate_send_recv_buffers

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
MaterialPresenceExchanger<dim, device_t>::pack_mirror_data(
  MaterialPresenceView_t userdata_material_presence,
  orchard_key_view_t     mirror_keys_device)
{
  auto amr_hashmap = m_mesh_map.hashmap();
  auto num_mirror_quad = m_amr_mesh.local_num_mirrors();

  [[maybe_unused]] auto local_num_quadrants = m_amr_mesh.local_num_quadrants();

  // make sure m_send buffer was allocated with the right size
  assertm(static_cast<size_t>(num_mirror_quad) == m_send.size(),
          "[MaterialPresenceExchanger::pack_mirror_data] send buffer has wrong size");

  {
    // traverse the list of mirror keys, and for each key try to find it in the old hashmap
    Kokkos::parallel_for(
      "pack_mirror_data",
      Kokkos::RangePolicy<exec_space>(0, num_mirror_quad),
      KOKKOS_CLASS_LAMBDA(const int32_t & imirror) {
        auto key = mirror_keys_device(imirror);

        // find key index in map
        auto key_index_map = amr_hashmap.find(key);

        // first check if key exists in the hashmap, if it exists, it means we found a matching
        // owned mirror quadrant
        if (amr_hashmap.valid_at(key_index_map))
        {
          auto userdata_index = static_cast<int32_t>(amr_hashmap.value_at(key_index_map));
          KOKKOS_ASSERT(
            userdata_index < local_num_quadrants &&
            "[MaterialPresenceExchanger::pack_mirror_data] index must correspond to a locally "
            "owned quadrant.");
          MaterialPresenceView_t::copy(m_send, imirror, userdata_material_presence, userdata_index);
        }
      });
  }

  Kokkos::fence();
} // pack_mirror_data

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
MaterialPresenceExchanger<dim, device_t>::do_mpi_send_recv()
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
        m_amr_mesh.ghost()->proc_offsets[iproc] * m_recv.block_length_per_octant(),
        (m_amr_mesh.ghost()->proc_offsets[iproc] + nghosts) * m_recv.block_length_per_octant());
      auto recv_buff = Kokkos::subview(m_recv.logical_view(), recv_range);

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
        m_amr_mesh.ghost()->mirror_proc_offsets[iproc] * m_send.block_length_per_octant(),
        (m_amr_mesh.ghost()->mirror_proc_offsets[iproc] + nmirrors) *
          m_send.block_length_per_octant());
      auto send_buff = Kokkos::subview(m_send.logical_view(), send_range);

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

// ===========================================================================
// ===========================================================================
template <size_t dim, typename device_t>
void
MaterialPresenceExchanger<dim, device_t>::unpack_ghost_data(
  MaterialPresenceView_t userdata_material_presence)
{
  // get the number of owned quadrants in local MPI process
  const auto num_owned_quad = m_amr_mesh.local_num_quadrants();

  // get the number of ghost quadrants in local MPI process
  const auto num_ghost_quad = m_amr_mesh.local_num_ghosts();

  // number of cells per block, and scalar values per cell
  const auto num_elts_per_quad = userdata_material_presence.block_length_per_octant();

  // make sure m_recv buffer was allocated with the right size
  assertm(static_cast<size_t>(num_ghost_quad) == m_recv.size(),
          "[MaterialPresenceExchanger::unpack_ghost_data] receive buffer has wrong size");

  auto recv_range = std::pair<std::size_t, std::size_t>(
    num_elts_per_quad * num_owned_quad, num_elts_per_quad * (num_owned_quad + num_ghost_quad));
  auto userdata_mp_ghost = Kokkos::subview(userdata_material_presence.logical_view(), recv_range);

  Kokkos::deep_copy(userdata_mp_ghost, m_recv.logical_view());

  Kokkos::fence();

} // unpack_ghost_data

// explicit template instantiation
template class MaterialPresenceExchanger<2, DefaultDevice>;
template class MaterialPresenceExchanger<3, DefaultDevice>;

} // namespace kalypsso
