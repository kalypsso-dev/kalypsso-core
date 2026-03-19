// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCells.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h> // for orchard_key_view_t alias
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>

namespace kalypsso
{

/**
 * \class FillBlockGhostCellsFunctor
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
 * block) is copied in the ghost e.g. right face ("o" symbol, ghost cells)
 *
 *                       . . . . . .
 *     x x x o           o x x x x .
 *     x x x o     ==>   o x x x x .
 *     x x x o           o x x x x .
 *     x x x o           o x x x x .
 *                       . . . . . .
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
class FillBlockGhostCellsFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  /**
   *
   * \param[in] stencil helper
   * \param[in] iOct_begin is the first octant to process
   * \param[in] num_octants is the number of octant to process
   * \param[in] userdata_in data array used to fill ghost of userdata_out
   * \param[in,out] userdata_out data array which we want to fill the block ghosts
   *                cells
   * \param[in] prolongation selects how coarse neighbor must be prolongated to fill ghost cells
   */
  FillBlockGhostCellsFunctor(StencilHelper_t              stencil_helper,
                             int32_t                      iOct_begin,
                             int32_t                      num_octants,
                             DataArrayBlock_t             userdata_in,
                             DataArrayGhostedBlock_t      userdata_out,
                             CellCenteredProlongationType prolongation)
    : m_stencil_helper(stencil_helper)
    , m_iOct_begin(iOct_begin)
    , m_num_octants(num_octants)
    , m_userdata_in(userdata_in)
    , m_userdata_out(userdata_out)
    , m_prolongation(prolongation)
    , m_cons_interpol()
  {
    KOKKOS_ASSERT(userdata_out.block_size() == userdata_in.block_size() &&
                  "userdata_in and userdata_out must have the same block sizes.");
  }

  // ====================================================================
  // ====================================================================
  static void
  check_args_validity(DataArrayBlock_t userdata_in, CellCenteredProlongationType prolongation_type);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        [[maybe_unused]] int32_t local_num_octants,
        int32_t                  iOct_begin,
        int32_t                  num_octants,
        DataArrayBlock_t         userdata_in,
        DataArrayGhostedBlock_t  userdata_out,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic);

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_neigh,
                                          coord_t<2> const &      coord_in,
                                          index_t const &         cellindex_out,
                                          int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_neigh,
                                          coord_t<3> const &      coord_in,
                                          index_t const &         cellindex_out,
                                          int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do second order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order2(CellLocation<2> const & cell_loc_neigh,
                                    coord_t<2> const &      coord_out,
                                    index_t const &         cellindex_out,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do second order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order2(CellLocation<3> const & cell_loc_neigh,
                                    coord_t<3> const &      coord_out,
                                    index_t const &         cellindex_out,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do fourth order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order4(CellLocation<2> const & cell_loc_neigh,
                                    coord_t<2> const &      coord_out,
                                    index_t const &         cellindex_out,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do fourth order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order4(CellLocation<3> const & cell_loc_neigh,
                                    coord_t<3> const &      coord_out,
                                    index_t const &         cellindex_out,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] cellindex_in is the cell index of the cell to read data from
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
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata_in;

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayGhostedBlock_t m_userdata_out;

  //! prolongation type
  const CellCenteredProlongationType m_prolongation;

  //! Conservative interpolation data
  const ConservativeInterpolation m_cons_interpol;

}; // class FillBlockGhostCellsFunctor

extern template class FillBlockGhostCellsFunctor<2, kalypsso::DefaultDevice>;
extern template class FillBlockGhostCellsFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTCELLS_H_
