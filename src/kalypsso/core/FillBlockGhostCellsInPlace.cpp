// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCellsInPlace.cpp
 * \brief \copybrief FillBlockGhostCellsInPlace.h
 */
#include <kalypsso/core/FillBlockGhostCellsInPlace.h>

namespace kalypsso
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillBlockGhostCellsInPlaceFunctor<dim, device_t>::apply(ConfigMap const &        config_map,
                                                        amr_hashmap_t            amr_hashmap,
                                                        orchard_key_view_t       orchard_keys,
                                                        DataArrayGhostedBlock_t  userdata,
                                                        brick_size_t<dim>        brick_sizes,
                                                        Kokkos::Array<bool, dim> is_brick_periodic)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata.block_size(), brick_sizes, is_brick_periodic);

  FillBlockGhostCellsInPlaceFunctor<dim, device_t> functor(
    stencil_helper, userdata, get_cell_prolongation_type(config_map));

  const auto nbCellsPerGhostedLeaf = userdata.num_cells();
  const auto nbCellsTotal = userdata.num_quadrants() * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for(
    "FillBlockGhostCellsInPlaceFunctor", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor);

} // FillBlockGhostCellsInPlaceFunctor<dim, device_t>::apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsInPlaceFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const & cell_loc_neigh,
  coord_t<2> const &      coord_in,
  index_t const &         cellindex_out,
  int32_t const &         iOct) const
{
  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for
  const auto   nbvar = m_userdata.num_vars();

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata, slope_type);

    // extrapolate
    m_userdata(cellindex_out, ivar, iOct) =
      m_userdata(cell_loc_neigh.ijk, ivar, cell_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx + ONE_FOURTH_F * static_cast<real_t>(iy) * dudy;
  }
} // FillBlockGhostCellsInPlaceFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsInPlaceFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const & cell_loc_neigh,
  coord_t<3> const &      coord_in,
  index_t const &         cellindex_out,
  int32_t const &         iOct) const
{
  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for
  const auto   nbvar = m_userdata.num_vars();

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;
  const int iz = 2 * (coord_in[IZ] - 2 * (coord_in[IZ] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata, slope_type);
    auto const dudz = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_z, cell_loc_left_z, ivar, m_userdata, slope_type);

    // extrapolate
    m_userdata(cellindex_out, ivar, iOct) =
      m_userdata(cell_loc_neigh.ijk, ivar, cell_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx +
      ONE_FOURTH_F * static_cast<real_t>(iy) * dudy + ONE_FOURTH_F * static_cast<real_t>(iz) * dudz;
  }

} // FillBlockGhostCellsInPlaceFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsInPlaceFunctor<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
                                                              coord_t<dim> const & coord_out,
                                                              int32_t const &      iOct) const
{

  const auto & b = m_userdata.block_size();

  coord_t<dim> coord_in;
  const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

  // int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_userdata.block_size());

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  if (dir_norm != 0)
  {
    // current cell is a ghost cell (thus overlapping with a neighbor block)

    /*
     * fill ghosts all around
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct);

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const CellLocation_t cell_loc_cur{ coord_in, key_cur, iOct, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    const auto nbvar = m_userdata.num_vars();

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     */
    const auto iOct_in = cell_loc_neigh.iOct;

    if (cell_loc_neigh.level() == cell_loc_cur.level())
    {

      // doing a simple copy (this is probably not need, cellindex_in computed above is ok)
      for (int32_t ivar = 0; ivar < nbvar; ++ivar)
      {
        m_userdata(cellindex_out, ivar, iOct) = m_userdata(cell_loc_neigh.ijk, ivar, iOct_in);
      }
    }
    else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser

      if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value
        for (int32_t ivar = 0; ivar < nbvar; ++ivar)
        {
          m_userdata(cellindex_out, ivar, iOct) = m_userdata(cell_loc_neigh.ijk, ivar, iOct_in);
        }
      }
      else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        linear_extrapolate_using_limited_slopes(cell_loc_neigh, coord_in, cellindex_out, iOct);
      }
    }
    else if (cell_loc_neigh.level() == cell_loc_cur.level() + 1)
    {
      // doing a restriction
      for (int32_t ivar = 0; ivar < nbvar; ++ivar)
      {
        m_userdata(cellindex_out, ivar, iOct) =
          m_stencil_helper.compute_siblings_average(cell_loc_neigh, ivar, m_userdata);

      } // for ivar
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm == 0)

} // FillBlockGhostCellsInPlaceFunctor<dim, device_t>::fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsInPlaceFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata.num_cells();

  const auto iOct = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct * nbCellsPerGhostedLeaf;

  const auto coord_out =
    cellindex_to_coord<dim>(cell_index_out, m_userdata.ghosted_block_size(), m_userdata.shift());

  fill_ghosts(cell_index_out, coord_out, iOct);

} // FillBlockGhostCellsInPlaceFunctor<dim, device_t>::operator()

template class FillBlockGhostCellsInPlaceFunctor<2, kalypsso::DefaultDevice>;
template class FillBlockGhostCellsInPlaceFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
