// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCellsInPlace.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTCELLSINPLACE_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTCELLSINPLACE_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h> // for orchard_key_view_t alias
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>

namespace kalypsso
{

/**
 * \class FillBlockGhostCellsInPlaceFunctor
 *
 * \sa FillBlockGhostCellsFunctor
 *
 * Same functionality as FillBlockGhostCellsFunctor but here the ghosted data array is used in input
 * (for the inner block of cells) and in output (to fill the ghost cells).
 *
 * This is a kokkos functor class that takes as input a DataArrayGhostedBlock<dim, device_t>
 * instance, and for each quadrant, fill block ghosts cells (across faces, edges and corners).
 *
 * This class only supports block sizes that are even integers.
 *
 * This class can be used in a piecewise loop over all octants.
 *
 * Just to illustrate, let's consider 4x4 blocks, with 1 ghost cell all around.
 * Below are 2 blocks (same ARM level); the right column of the left block (symbol "o") is copied
 * into the ghost of the right block, and vice-versa with symbol 'r'
 *
 *   . . . . . .           . . . . . .
 *   . x x x o r           o r x x x .
 *   . x x x o r   <==>    o r x x x .
 *   . x x x o r           o r x x x .
 *   . x x x o r           o r x x x .
 *   . . . . . .           . . . . . .
 *
 *
 */
template <size_t dim, typename device_t>
class FillBlockGhostCellsInPlaceFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  /**
   *
   * \param[in] stencil helper
   * \param[in] iOct_begin is the first octant to process
   * \param[in] num_octants is the number of octant to process
   * \param[in,out] userdata data array which we want to fill the block ghosts
   *                cells
   * \param[in] prolongation selects how coarse neighbor must be prolongated to fill ghost cells
   */
  FillBlockGhostCellsInPlaceFunctor(StencilHelper_t              stencil_helper,
                                    DataArrayGhostedBlock_t      userdata,
                                    CellCenteredProlongationType prolongation)
    : m_stencil_helper(stencil_helper)
    , m_userdata(userdata)
    , m_prolongation(prolongation)
  {}

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        DataArrayGhostedBlock_t  userdata,
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
                                          int32_t const &         iOct) const;

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
                                          int32_t const &         iOct) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(index_t const &      cellindex_out,
              coord_t<dim> const & coord_out,
              int32_t const &      iOct) const;

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

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayGhostedBlock_t m_userdata;

  //! prolongation type
  const CellCenteredProlongationType m_prolongation;

}; // class FillBlockGhostCellsInPlaceFunctor

extern template class FillBlockGhostCellsInPlaceFunctor<2, kalypsso::DefaultDevice>;
extern template class FillBlockGhostCellsInPlaceFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTCELLSINPLACE_H_
