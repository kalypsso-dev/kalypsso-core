// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file NeighborConnectivity.h
 *
 * Kokkos functor which computes the number of neighbor quadrants of each quadrants (either
 * owned+ghost or only mirror quadrants).
 */
#ifndef KALYPSSO_CORE_NEIGHBORCONNECTIVITY_H_
#define KALYPSSO_CORE_NEIGHBORCONNECTIVITY_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/mesh_utils.h> // for definition of Face::XMIN, etc...
#include <kalypsso/core/utils_direction.h>

#include <kalypsso/core/NeighborLocalOffsets.h>
#include <kalypsso/core/utils_sector.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <../better-enums/enum.h>
#include <type_traits>

namespace kalypsso
{

namespace core
{

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class NeighborConnectivity
 *
 * This functor computer the number of neighbors of a quadrant using two variants:
 * - compute the exact number of neighbors
 * - compute a slightly over-estimate number of neighbor (the coarse neighbor maybe counted x2 in 2d
 * or 4x in 3d)
 *
 * Three type s of information are stored in this class:
 * - an array of the number of direct neighbors stored as an 8-bit unsigned integer, one per
 * quadrant.
 * - an array of local offsets to number of direct neighbors per sector bits; as there are 9 sectors
 * in 2d and 27 sectors in 3d, local offsets are stored using a single 64 bit integer in 2D and 3
 * 64-bits integers in 3d
 * - global offsets to where the neighbors of a given quadrant are stored; stored as a 64 bit
 * integer and computed as cumulative scan of the number of neighbor per quadrant (first array
 * above)
 *
 * Counting the number of neighbor is necessary for memory allocation for having a neighbor
 * connectivity table.
 */
template <size_t dim, typename device_t>
class NeighborConnectivity
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  /**
   * typename used to represent the number of neighbors of a given quadrant.
   *
   * the worst case is obtained when a quadrant is surrounded by neighbors at larger AMR level:
   * - in 2d, that is 12 neighbors (4 corner neighbors + 2x4 face neighbors)
   * - in 3d, that is 56 neighbors (8 corner neighbors + 4*6 face neighbors + 2*12 edge neighbors)
   *
   * So 6 bits are enough to store this information, in practice we use a 8 bit integer to avoid
   * unnecessary bit manipulation.
   */
  using neighbor_count_t = uint8_t;
  using neighbor_count_view_t = DataArray<neighbor_count_t, device_t>;

  /**
   * typename used to store local offsets to neighbors rank by sector
   */
  using local_offsets_t = typename NeighborLocalOffsets<dim>::local_offsets_t;
  using local_offsets_view_t = DataArray<local_offsets_t, device_t>;

  /**
   * typename used to store global offsets in the list of neighbors to indiquate where
   * where the neighbors of a given quadrant start
   */
  using global_offsets_t = uint64_t;
  using global_offsets_view_t = DataArray<global_offsets_t, device_t>;

  /**
   * typename to store neighbor octant ids
   */
  using neighbor_octant_ids_view_t = DataArray<iOct_t, device_t>;

  //! Tag class used in operator() for computing the number of neighbors of each quadrant
  struct TagCountNeighbors
  {};

  //! Tag class used in operator() for filling the connectivity table
  struct TagFillNeighbors
  {};

private:
  //! parallel environment
  ParallelEnvDevice m_par_env;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! Neighbor count array.
  //! How many neighbors per quadrant in a 3x3 neighborhood
  neighbor_count_view_t m_neighbor_count_view;

  //! Array of encoded local offsets (same size as neighbor count view).
  //! Local offsets are defined as accumulated number of neighbors in all preceding sectors.
  //! Sectors are defined in NeighborLocalOffsets
  local_offsets_view_t m_local_offsets_view;

  //! Array of global offsets (same size as neighbor count view + 1).
  //! Global offsets are offsets into the list of neighbors, one offset for each quadrant
  //! (should be computed as a cumulative scan of the neighbor count view)
  global_offsets_view_t m_global_offsets_view;

  //! Array of neighbor octant ids (this is the connectivity table)
  //! the size of this view is compute as the sum of neighbor count view elements (actually obtained
  //! via an exclusive scan as need also need global offsets)
  neighbor_octant_ids_view_t m_neighbor_octant_ids_view;

public:
  /**
   * Constructor.
   *
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
   *            each dimension)
   * \param[in] is_brick_periodic array of boolean value indicating if the p4est
   *            brick connectivity is periodic in the given dimension
   * \param[in] type determines which count method is used
   */
  NeighborConnectivity(ParallelEnv const &        par_env,
                       amr_hashmap_t const &      amr_hashmap,
                       orchard_key_view_t const & orchard_keys,
                       brick_size_t<dim>          brick_sizes,
                       Kokkos::Array<bool, dim>   is_brick_periodic)
    : m_par_env(par_env)
    , m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_neighbor_count_view("neighbor count view", orchard_keys.extent(0))
    , m_local_offsets_view("local offsets view", orchard_keys.extent(0))
    , m_global_offsets_view("global offsets view", orchard_keys.extent(0) + 1)
    , m_neighbor_octant_ids_view("neighbor octant ids view", 0)
  {}

  // ==============================================================
  // ==============================================================
  //! Refresh neighbor connectivity
  static void
  refresh_neighbor_count(NeighborConnectivity & neighbor_connectivity);

  // =========================================================================
  // =========================================================================
  size_t
  num_quadrants()
  {
    return m_orchard_keys_device.extent(0);
  }

  // =========================================================================
  // =========================================================================
  auto
  neighbor_count()
  {
    return m_neighbor_count_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  neighbor_count() const
  {
    return m_neighbor_count_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  global_offsets()
  {
    return m_global_offsets_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  global_offsets() const
  {
    return m_global_offsets_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  neighbor_octants_ids()
  {
    return m_neighbor_octant_ids_view;
  }

  // =========================================================================
  // =========================================================================
  auto
  neighbor_octants_ids() const
  {
    return m_neighbor_octant_ids_view;
  }

  // =========================================================================
  // =========================================================================
  void
  resize()
  {
    m_neighbor_count_view.resize(m_orchard_keys_device.size());
    m_local_offsets_view.resize(m_orchard_keys_device.size());
    m_global_offsets_view.resize(m_orchard_keys_device.size() + 1);
  }

  // =========================================================================
  // =========================================================================
  void
  resize_connectivity(size_t total_number_of_neighbors)
  {
    m_neighbor_octant_ids_view.resize(total_number_of_neighbors);
  }

  // =========================================================================
  // =========================================================================
  uint64_t
  allocated_size_in_bytes()
  {
    uint64_t total = 0;
    total += m_neighbor_count_view.allocated_size_in_bytes();
    total += m_local_offsets_view.allocated_size_in_bytes();
    total += m_global_offsets_view.allocated_size_in_bytes();
    total += m_neighbor_octant_ids_view.allocated_size_in_bytes();
    return total;
  }

  // =========================================================================
  // =========================================================================
  void
  report(ParallelEnv const & par_env, std::string type);

private:
  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION auto
  count_face_neighbors(Face::face_t face, key_t key_cur) const -> neighbor_count_t;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  auto
  count_edge_neighbors(Face::face_t face0, Face::face_t face1, key_t key_cur) const
    -> neighbor_count_t;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION auto
  count_corner_neighbors(Face::face_t face0,
                         Face::face_t face1,
                         Face::face_t face2,
                         key_t        key_cur) const -> neighbor_count_t;

  // ==============================================================
  // ==============================================================
  /**
   * Count neighbors in a given sector.
   *
   * This is just a wrapper around face, edge and corner count.
   *
   * Here is the sector enumeration in 2d
   *
   *               |                     |
   *  sector 6     |     sector 7        |  sector 8
   *          _____|_____________________|_______
   *               |                     |
   *               |                     |
   *               |                     |
   *               |                     |
   *  sector 3     |     sector 4        |  sector 5
   *               |                     |
   *               |                     |
   *               |                     |
   *               |                     |
   *          _____|_____________________|_______
   *               |                     |
   *  sector 0     |     sector 1        | sector 2
   *               |                     |
   */
  KOKKOS_INLINE_FUNCTION auto
  count_sector_neighbors(uint8_t sector, key_t key_cur) const -> neighbor_count_t;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_face_neighbors(Face::face_t     face,
                      global_offsets_t global_offset,
                      uint8_t          local_offset,
                      key_t            key_cur) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void
  fill_edge_neighbors(Face::face_t     face0,
                      Face::face_t     face1,
                      global_offsets_t global_offset,
                      uint8_t          local_offset,
                      key_t            key_cur) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_corner_neighbors(Face::face_t     face0,
                        Face::face_t     face1,
                        Face::face_t     face2,
                        global_offsets_t global_offset,
                        uint8_t          local_offset,
                        key_t            key_cur) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_sector_neighbors(uint8_t sector, key_t key_cur, index_t iOct) const;

public:
  // ==============================================================
  // ==============================================================
  /**
   * Compute the number of neighbors of each quadrant.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagCountNeighbors const &, const index_t & iOct) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill the connectivity table.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFillNeighbors const &, const index_t & iOct) const;

}; // class NeighborConnectivity

// explicit template instantiation
extern template class NeighborConnectivity<2, kalypsso::DefaultDevice>;
extern template class NeighborConnectivity<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_NEIGHBORCONNECTIVITY_H_
