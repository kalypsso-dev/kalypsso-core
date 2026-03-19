// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCells_v2.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_V2_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_V2_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h> // for orchard_key_view_t alias
#include <kalypsso/core/amr_hashmap.h>

namespace kalypsso
{

/**
 * \class FillBlockGhostCellsFunctorV2
 *
 * This is a kokkos functor class that takes as input a DataArrayGhostedBlock<dim, device_t>
 * instance, and for each quadrant, fill block ghosts cells (across faces, edges and corners).
 * The input array is non-ghosted, output array is ghosted.
 *
 * This class only supports block sizes that are even integers.
 *
 * This class can be used in a piecewise loop over leaf octant.
 *
 * Just to illustrate, let's consider 4x4 blocks, with 1 ghost cell all around.
 * Below are 2 blocks (same ARM level); the right column of the left block (symbol "o", non-ghosted
 * block) is copied in the ghost e.g. right face ("o" symbol, ghosted blocked)
 *
 *   . . . . . .         . . . . . .
 *   . x x x o .         o x x x x .
 *   . x x x o .   ==>   o x x x x .
 *   . x x x o .         o x x x x .
 *   . x x x o .         o x x x x .
 *   . . . . . .         . . . . . .
 *
 * The main difficulty here is to deploy the entire combinatorics of
 * geometrical possibilities in terms of
 * - size of neighbor octant, i.e.
 *   is neighbor octant small, same size or larger than current octant,
 * - direction : face along X, Y or Z behave slightly differently, need to
 *   efficiently take symetries into account
 * - 2d/3d
 *
 * So we need to be careful, have good testing code.
 * See file test_AMRmesh_fill_block_ghosts.cpp
 *
 */
template <size_t dim, typename device_t>
class FillBlockGhostCellsFunctorV2
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  /**
   *
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] iOct_begin is the first octant to process
   * \param[in] num_octants is the number of octant to process
   * \param[in] userdata_in data array used to fill ghost of userdata_out
   * \param[in,out] userdata_out data array which we want to fill the block ghosts
   *                cells
   * \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
   *            each dimension)
   * \param[in] is_brick_periodic array of boolean value indicating if the p4est
   *            brick connectivity is periodic in the given dimension
   *
   */
  FillBlockGhostCellsFunctorV2(amr_hashmap_t            amr_hashmap,
                               orchard_key_view_t       orchard_keys,
                               int32_t                  iOct_begin,
                               int32_t                  num_octants,
                               DataArrayBlock_t         userdata_in,
                               DataArrayGhostedBlock_t  userdata_out,
                               brick_size_t<dim>        brick_sizes,
                               Kokkos::Array<bool, dim> is_brick_periodic);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        int32_t                  local_num_octants,
        int32_t                  iOct_begin,
        int32_t                  num_octants,
        DataArrayBlock_t         userdata_in,
        DataArrayGhostedBlock_t  userdata_out,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic);

  // ==============================================================
  // ==============================================================
  /**
   * compute siblings average.
   *
   * In other words, given the coordinate of a given cell (all even integer, so that it
   * corresponds to the eldest sibling of a block 2^dim cells), then we perform the average of a
   * given field of m_userdata_in over all siblings.
   *   ____________
   *  |     |     |
   *  |     |     |
   *  |_____|_____|
   *  |     |     |
   *  |  X  |     |
   *  |_____|_____|
   *
   *
   * \param[in] coordinates of the eldest sibling (must be all even integers).
   * \param[in] ivar variable id
   * \param[in] iOct octant id
   *
   * \return average value
   */
  KOKKOS_INLINE_FUNCTION real_t
  compute_siblings_average(coord_t<dim> const & coords,
                           int32_t const &      ivar,
                           int32_t const &      iOct) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] cellindex_out is the cell index of the ghost cell to fill
   * \param[in] iOct_global is the octant id among all octant owned by current MPI process.
   *
   * Just to be clear iOct_global - m_iOct_begin is the local octant id inside the group of octant
   * being processed.
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_inner(int32_t cellindex_in, int32_t cellindex_out, int32_t iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at the same AMR level.
   *
   * \param[in] iOct_global index to current octant
   * \param[in] iOct_neigh index to neighbor octant
   * \param[in] cell index integer used to map the ghost cell to fill
   * \param[in] dir is array of direction pointing to neighbor
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_ghost_same_level(int32_t iOct_global,
                        int32_t iOct_neigh,
                        index_t cellindex_in,
                        index_t cellindex_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at coarser level.
   *
   * \param[in] key is current octant orchard key
   * \param[in] key_neigh is neighbor octant orchard key
   * \param[in] child_id is the child id of neighbor (at same level) wrt actual neighbor (coarser
   *            level)
   * \param[in] iOct_global index to current octant
   * \param[in] iOct_neigh index to neighbor octant
   * \param[in] coord_in
   * \param[in] cellindex_out integer used to map the ghost cell to fill
   * \param[in] dir is direction to neighbor (in a 3x3 neighborhood)
   *
   * In 2d, for face neighbors, there are 2 typical situations :
   *  ______               ______    __
   * |      |             |      |  X  |
   * |      |   __    or  |      |  X__|
   * |      |  X  |       |      |
   * |______|  X__|       |______|
   *
   * In this function, we want to fill the "X" ghost cells using data from
   * the (larger) neighbor octant.
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_ghost_coarser_level(uint8_t      child_id,
                           int32_t      iOct_global,
                           int32_t      iOct_neigh,
                           coord_t<dim> coord_in,
                           index_t      cellindex_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at finer level.
   *
   * \param[in] key_neigh_same_level is orchard key of neighbor in direction dir at same AMR level
   * \param[in] iOct_global is index to current octant
   * \param[in] cellindex integer used to map the ghost cell to fill
   * \param[in] dir is direction to neighbor (in a 3x3 neighborhood)
   *
   * current  octant (large one) is on the right
   * neighbor octant (small one) is on the left
   *
   * In 2d, for face neighbor, there are 2 typical situations :
   *        _______             __      _______
   *       |       |           |  |    X       |
   *  __   |       |      or   |__|    X       |
   * |  |  X       |                   |       |
   * |__|  x_______|                   |_______|
   *
   */
  KOKKOS_INLINE_FUNCTION
  bool
  fill_ghost_finer_level(key_t        key_neigh_same_level,
                         int32_t      iOct_global,
                         coord_t<dim> coord_in,
                         index_t      cellindex_out) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(index_t const &      cellindex_out,
              coord_t<dim> const & coord_out,
              int32_t const &      iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index) const;

private:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata_in;

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayGhostedBlock_t m_userdata_out;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

}; // class FillBlockGhostCellsFunctorV2

extern template class FillBlockGhostCellsFunctorV2<2, kalypsso::DefaultDevice>;
extern template class FillBlockGhostCellsFunctorV2<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_V2_H_
