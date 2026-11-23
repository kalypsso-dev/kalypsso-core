// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeNumberOfNeighborsOfQuadrant.h
 *
 * Kokkos functor which computes the number of neighbor quadrants of each quadrants (either
 * owned+ghost or only mirror quadrants).
 */
#ifndef KALYPSSO_CORE_COMPUTENUMBEROFNEIGHBORSOFQUADRANT_H_
#define KALYPSSO_CORE_COMPUTENUMBEROFNEIGHBORSOFQUADRANT_H_

#include <atomic>
#include <kalypsso/core/DataArray.h>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/mesh_utils.h> // for definition of Face::XMIN, etc...
#include <kalypsso/core/utils_direction.h>

#include <../better-enums/enum.h>
#include <type_traits>

namespace kalypsso
{

namespace core
{
// clang-format off
/**
 * An enum type to tell if the exact or approximated number of neighbors should be computed.
 */
BETTER_ENUM(ComputeNumberOfNeighborsOfQuadrantType, uint8_t,
            APPROXIMATED = 0,
            EXACT = 1)
// clang-format on


/**
 * \class ComputeNumberOfNeighborsOfQuadrant
 *
 * This functor computer the number of neighbors of a quadrant using two variants:
 * - compute the exact number of neighbors
 * - compute a slightly over-estimate number of neighbor (the coarse neighbor maybe counted x2 in 2d
 * or 4x in 3d)
 *
 * The purpose is to do memory allocation for a neighbor connectivity table.
 */
template <size_t dim, typename device_t>
class ComputeNumberOfNeighborsOfQuadrant
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

private:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! neighbor count array
  neighbor_count_view_t m_neighbor_count_view;

  //! How to count number of neighbor (approximate value or exact value)
  ComputeNumberOfNeighborsOfQuadrantType m_count_type;

public:
  /**
   * Constructor.
   *
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_mesh_info (number of owned, MPI ghosts and outside quadrants)
   * \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
   *            each dimension)
   * \param[in] is_brick_periodic array of boolean value indicating if the p4est
   *            brick connectivity is periodic in the given dimension
   * \param[out] neighbor_count_view
   * \param[in] type determines which count method is used (approximate or exact)
   */
  ComputeNumberOfNeighborsOfQuadrant(amr_hashmap_t                          amr_hashmap,
                                     orchard_key_view_t                     orchard_keys,
                                     AMRMeshInfo                            amr_mesh_info,
                                     brick_size_t<dim>                      brick_sizes,
                                     Kokkos::Array<bool, dim>               is_brick_periodic,
                                     neighbor_count_view_t                  neighbor_count_view,
                                     ComputeNumberOfNeighborsOfQuadrantType type)
    : m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_amr_mesh_info(amr_mesh_info)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_neighbor_count_view(neighbor_count_view)
    , m_count_type(type)
  {}

  // ==============================================================
  // ==============================================================
  //! Count number of neighbors.
  static auto
  count_neighbors(amr_hashmap_t                          amr_hashmap,
                  orchard_key_view_t                     orchard_keys,
                  AMRMeshInfo                            amr_mesh_info,
                  brick_size_t<dim>                      brick_sizes,
                  Kokkos::Array<bool, dim>               is_brick_periodic,
                  ComputeNumberOfNeighborsOfQuadrantType type) -> neighbor_count_view_t;

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
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & iOct) const;

}; // class ComputeNumberOfNeighborsOfQuadrant

// explicit template instantiation
extern template class ComputeNumberOfNeighborsOfQuadrant<2, kalypsso::DefaultDevice>;
extern template class ComputeNumberOfNeighborsOfQuadrant<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTENUMBEROFNEIGHBORSOFQUADRANT_H_
