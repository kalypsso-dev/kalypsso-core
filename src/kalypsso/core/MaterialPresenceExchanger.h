// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MaterialPresenceExchanger.h
 */

#ifndef KALYPSSO_CORE_MATERIALPRESENCEEXCHANGER_H_
#define KALYPSSO_CORE_MATERIALPRESENCEEXCHANGER_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost
#include <kalypsso/core/MaterialPresence.h>

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
 * \class MaterialPresenceExchanger
 *
 * \brief Does the same thing as MeshGhostsExchanger but specialized for material presence.
 */
template <size_t dim, typename device_t>
class MaterialPresenceExchanger
{
public:
  using exec_space = typename device_t::execution_space;

  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  using comm_buffer_t = MaterialPresenceView_t;

  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

  // =========================================================================
  // =========================================================================
  //! constructor
  MaterialPresenceExchanger(const ConfigMap &        config_map,
                            const ParallelEnv &      par_env,
                            AMRmesh<dim> &           amr_mesh,
                            MeshMap<dim, device_t> & mesh_map)
    : m_config_map(config_map)
    , m_par_env(par_env)
    , m_amr_mesh(amr_mesh)
    , m_mesh_map(mesh_map)
    , m_send("MaterialPresenceExchanger::allocate send",
             static_cast<uint32_t>(config_map.getInteger("run", "nmat", 1)),
             0)
    , m_recv("MaterialPresenceExchanger::allocate recv",
             static_cast<uint32_t>(config_map.getInteger("run", "nmat", 1)),
             0)
  {}

  // =========================================================================
  // =========================================================================
  //! destructor
  ~MaterialPresenceExchanger() = default;

  // =========================================================================
  // =========================================================================
  //! do it all, pack, exchange, unpack.
  //!
  //! \param[in] userdata_block is a userdata block array (i.e. one value per face); must
  //!            be sized upon the total number of octant in current MPI process (owned + ghost)
  void
  exchange(MaterialPresenceView_t userdata_material_presence);

  // =========================================================================
  // =========================================================================
  auto
  allocated_size_in_bytes() const -> size_t;

  // =========================================================================
  // =========================================================================
  //! allocate device buffer for performing ghost data exchange
  //!
  void
  allocate_send_recv_buffers();

  // =========================================================================
  // =========================================================================
  //! pack mirror data into send buffer (device function) - block (cell-center) version.
  //!
  //! implementation use array of orchard key (mirror quads) to address userdata and copy data into
  //! send buffer.
  //!
  //! \param[in] userdata_material_presence is supposed to be of size (forest->local_num_quadrants +
  //!            ghost->ghosts>elem_count)
  //! \param[in] mirror_keys_device orchard keys of mirror quadrant
  //!
  //! IMPORTANT NOTE: remember that orchard keys in mirror array are sorted first by MPI processor
  //! to send to and secondly by Morton order.
  //!
  //! make sure that MeshMap object is up-to-date before using this; we need the unordered map
  //! (orchard keys, index) to be valid, i.e. calling MeshMap::fill_map is necessary after any mesh
  //! changes (p4est_balance, p4est_partition)
  void
  pack_mirror_data(MaterialPresenceView_t userdata_material_presence,
                   orchard_key_view_t     mirror_keys_device);

  // =========================================================================
  // =========================================================================
  //! actually perform MPI comm to send and receive ghost quadrant userdata
  void
  do_mpi_send_recv();

  // =========================================================================
  // =========================================================================
  //! unpack received data (recv buffer) into ghost data (cell-center block).
  //!
  //! we will fill all ghost quadrant data, i.e. from index forest->local_num_quad
  //! to index forest->local_num_quadrants + ghost->ghosts>elem_count
  //!
  //! \param[in] MaterialPresenceView_t userdata_material_presence is supposed to be of size
  //!            (forest->local_num_quadrants + ghost->ghosts->elem_count)
  //!
  void
  unpack_ghost_data(MaterialPresenceView_t userdata_material_presence);

  // =========================================================================
  // =========================================================================
private:
  //! config map (input parameter)
  const ConfigMap & m_config_map;

  //! parallel environment
  const ParallelEnv & m_par_env;

  //! AMRmesh reference object, mostly for accessing p4est ghost object
  AMRmesh<dim> & m_amr_mesh;

  //! a MeshMap object so that we can extract the orchard keys as a kokkos view or unordered map
  MeshMap<dim, device_t> & m_mesh_map;

  //! send buffers
  comm_buffer_t m_send;

  //! recv buffers
  comm_buffer_t m_recv;

}; // class MaterialPresenceExchanger

// explicit template instantiation
extern template class MaterialPresenceExchanger<2, DefaultDevice>;
extern template class MaterialPresenceExchanger<3, DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_MATERIALPRESENCEEXCHANGER_H_
