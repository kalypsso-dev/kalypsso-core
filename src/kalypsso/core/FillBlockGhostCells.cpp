// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCells.cpp
 * \brief \copybrief FillBlockGhostCells.h
 */
#include <kalypsso/core/FillBlockGhostCells.h>

namespace kalypsso
{
// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FillBlockGhostCellsFunctor<dim, device_t>::check_args_validity(
  DataArrayBlock_t             userdata_in,
  CellCenteredProlongationType prolongation_type)
{
  if (prolongation_type == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_2)
  {
    {
      const auto & b = userdata_in.block_size();
      bool         invalid_size = b[IX] < 4 or b[IY] < 4;
      if constexpr (dim == 3)
      {
        invalid_size = invalid_size or b[IZ] < 5;
      }
      if (invalid_size)
        Kokkos::abort("Userdata block size is must have at least 4 cells in direction");
    }
  }
  if (prolongation_type == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_4)
  {
    {
      const auto & b = userdata_in.block_size();
      bool         invalid_size = b[IX] < 6 or b[IY] < 6;
      if constexpr (dim == 3)
      {
        invalid_size = invalid_size or b[IZ] < 6;
      }
      if (invalid_size)
        Kokkos::abort("Userdata block size is must have at least 6 cells in direction");
    }
  }
} // FillBlockGhostCellsFunctor<dim, device_t>::check_args_validity

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillBlockGhostCellsFunctor<dim, device_t>::apply(ConfigMap const &        config_map,
                                                 amr_hashmap_t            amr_hashmap,
                                                 orchard_key_view_t       orchard_keys,
                                                 [[maybe_unused]] int32_t local_num_octants,
                                                 int32_t                  iOct_begin,
                                                 int32_t                  num_octants,
                                                 DataArrayBlock_t         userdata_in,
                                                 DataArrayGhostedBlock_t  userdata_out,
                                                 brick_size_t<dim>        brick_sizes,
                                                 Kokkos::Array<bool, dim> is_brick_periodic)
{

  // make sure the range of octants to process is valid
  KOKKOS_ASSERT(iOct_begin + num_octants <= local_num_octants &&
                "Invalid range of octants to process");

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  const auto prolongation_type = get_cell_prolongation_type(config_map);

  check_args_validity(userdata_in, prolongation_type);

  FillBlockGhostCellsFunctor<dim, device_t> functor(
    stencil_helper, iOct_begin, num_octants, userdata_in, userdata_out, prolongation_type);

  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_octants * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for(
    "FillBlockGhostCellsFunctor", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor);

} // FillBlockGhostCellsFunctor<dim, device_t>::apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const & cell_loc_neigh,
  coord_t<2> const &      coord_in,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for
  const auto   nbvar = m_userdata_in.num_vars();

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in {-1, +1}
  //
  // --------------------------
  // |           |            |
  // |           |            |
  // |   -1,1    |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   -1,-1   |    1,-1    |
  // |           |            |
  // |___________|____________|
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);

    // extrapolate
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(
        cell_loc_neigh.cellindex(m_userdata_in.block_size()), ivar, cell_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx + ONE_FOURTH_F * static_cast<real_t>(iy) * dudy;
  }
} // FillBlockGhostCellsFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const & cell_loc_neigh,
  coord_t<3> const &      coord_in,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for
  const auto   nbvar = m_userdata_in.num_vars();

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
  // in {-1, +1}
  //
  // --------------------------
  // |           |            |
  // |           |            |
  // |   -1,1    |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   -1,-1   |    1,-1    |
  // |           |            |
  // |___________|____________|
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;
  const int iz = 2 * (coord_in[IZ] - 2 * (coord_in[IZ] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);
    auto const dudz = m_stencil_helper.compute_minmod_slopes(
      cell_loc_neigh, cell_loc_right_z, cell_loc_left_z, ivar, m_userdata_in, slope_type);

    // extrapolate
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(
        cell_loc_neigh.cellindex(m_userdata_in.block_size()), ivar, cell_loc_neigh.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx +
      ONE_FOURTH_F * static_cast<real_t>(iy) * dudy + ONE_FOURTH_F * static_cast<real_t>(iz) * dudz;
  }

} // FillBlockGhostCellsFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order2(
  CellLocation<2> const & cell_loc_neigh,
  coord_t<2> const &      coord_out,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdata_in.num_vars();

  // determine local position of current cell inside virtual parent cell using integer
  //  coordinates in {0, 1}
  //
  // --------------------------
  // |           |            |
  // |           |            |
  // |   0,1     |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   0,0     |    1,0     |
  // |           |            |
  // |___________|____________|
  const int ix = (coord_out[IX] - 2 * (coord_out[IX] / 2));
  const int iy = (coord_out[IY] - 2 * (coord_out[IY] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t val[3];

    for (int j = -1; j < 2; ++j)
    {
      // interpolate along X
      auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -1, j });
      auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 0, j });
      auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 1, j });

      val[j + 1] = m_stencil_helper.compute_linear_combination(cell_loc_x_m1,
                                                               cell_loc_x_0,
                                                               cell_loc_x_p1,
                                                               cell_loc_neigh.level(),
                                                               ivar,
                                                               m_userdata_in,
                                                               ix == 0 ? m_cons_interpol.COEFS2_L
                                                                       : m_cons_interpol.COEFS2_R);
    }

    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_cons_interpol.order2(val[0], val[1], val[2], iy == 0);

  } // end for ivar

} // FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order2 - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order2(
  CellLocation<3> const & cell_loc_neigh,
  coord_t<3> const &      coord_out,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdata_in.num_vars();

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in {0, 1}
  // --------------------------
  // |           |            |
  // |           |            |
  // |   0,1     |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   0,0     |    1,0     |
  // |           |            |
  // |___________|____________|
  const int ix = (coord_out[IX] - 2 * (coord_out[IX] / 2));
  const int iy = (coord_out[IY] - 2 * (coord_out[IY] / 2));
  const int iz = (coord_out[IZ] - 2 * (coord_out[IZ] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t valx[3][3];

    for (int k = -1; k < 2; ++k)
    {
      for (int j = -1; j < 2; ++j)
      {
        // interpolate along X
        auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -1, j, k });
        auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 0, j, k });
        auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 1, j, k });

        valx[k + 1][j + 1] = m_stencil_helper.compute_linear_combination(
          cell_loc_x_m1,
          cell_loc_x_0,
          cell_loc_x_p1,
          cell_loc_neigh.level(),
          ivar,
          m_userdata_in,
          ix == 0 ? m_cons_interpol.COEFS2_L : m_cons_interpol.COEFS2_R);
      }
    }
    real_t valy[3];

    valy[0] = m_cons_interpol.order2(valx[0][0], valx[0][1], valx[0][2], iy == 0);
    valy[1] = m_cons_interpol.order2(valx[1][0], valx[1][1], valx[1][2], iy == 0);
    valy[2] = m_cons_interpol.order2(valx[2][0], valx[2][1], valx[2][2], iy == 0);

    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_cons_interpol.order2(valy[0], valy[1], valy[2], iz == 0);

  } // end for ivar

} // FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order2 - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order4(
  CellLocation<2> const & cell_loc_neigh,
  coord_t<2> const &      coord_out,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdata_in.num_vars();

  // determine local position of current cell inside virtual parent cell using integer
  //  coordinates in {0, 1}
  //
  // --------------------------
  // |           |            |
  // |           |            |
  // |   0,1     |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   0,0     |    1,0     |
  // |           |            |
  // |___________|____________|
  const int ix = (coord_out[IX] - 2 * (coord_out[IX] / 2));
  const int iy = (coord_out[IY] - 2 * (coord_out[IY] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t val[5];

    for (int j = -2; j < 3; ++j)
    {
      // interpolate along X
      auto cell_loc_x_m2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -2, j });
      auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -1, j });
      auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 0, j });
      auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 1, j });
      auto cell_loc_x_p2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 2, j });

      val[j + 2] = m_stencil_helper.compute_linear_combination(cell_loc_x_m2,
                                                               cell_loc_x_m1,
                                                               cell_loc_x_0,
                                                               cell_loc_x_p1,
                                                               cell_loc_x_p2,
                                                               cell_loc_neigh.level(),
                                                               ivar,
                                                               m_userdata_in,
                                                               ix == 0 ? m_cons_interpol.COEFS4_L
                                                                       : m_cons_interpol.COEFS4_R);
    }

    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_cons_interpol.order4(val[0], val[1], val[2], val[3], val[4], iy == 0);

  } // end for ivar

} // FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order4 - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order4(
  CellLocation<3> const & cell_loc_neigh,
  coord_t<3> const &      coord_out,
  index_t const &         cellindex_out,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdata_in.num_vars();

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in {0, 1}
  // --------------------------
  // |           |            |
  // |           |            |
  // |   0,1     |    1,1     |
  // |           |            |
  // |___________|____________|
  // |           |            |
  // |           |            |
  // |   0,0     |    1,0     |
  // |           |            |
  // |___________|____________|
  const int ix = (coord_out[IX] - 2 * (coord_out[IX] / 2));
  const int iy = (coord_out[IY] - 2 * (coord_out[IY] / 2));
  const int iz = (coord_out[IZ] - 2 * (coord_out[IZ] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t valx[5][5];

    for (int k = -2; k < 3; ++k)
    {
      for (int j = -2; j < 3; ++j)
      {
        // interpolate along X
        auto cell_loc_x_m2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -2, j, k });
        auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -1, j, k });
        auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 0, j, k });
        auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 1, j, k });
        auto cell_loc_x_p2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 2, j, k });

        valx[k + 2][j + 2] = m_stencil_helper.compute_linear_combination(
          cell_loc_x_m2,
          cell_loc_x_m1,
          cell_loc_x_0,
          cell_loc_x_p1,
          cell_loc_x_p2,
          cell_loc_neigh.level(),
          ivar,
          m_userdata_in,
          ix == 0 ? m_cons_interpol.COEFS4_L : m_cons_interpol.COEFS4_R);
      }
    }
    real_t valy[5];

    valy[0] =
      m_cons_interpol.order4(valx[0][0], valx[0][1], valx[0][2], valx[0][3], valx[0][4], iy == 0);
    valy[1] =
      m_cons_interpol.order4(valx[1][0], valx[1][1], valx[1][2], valx[1][3], valx[1][4], iy == 0);
    valy[2] =
      m_cons_interpol.order4(valx[2][0], valx[2][1], valx[2][2], valx[2][3], valx[2][4], iy == 0);
    valy[3] =
      m_cons_interpol.order4(valx[3][0], valx[3][1], valx[3][2], valx[3][3], valx[3][4], iy == 0);
    valy[4] =
      m_cons_interpol.order4(valx[4][0], valx[4][1], valx[4][2], valx[4][3], valx[4][4], iy == 0);

    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_cons_interpol.order4(valy[0], valy[1], valy[2], valy[3], valy[4], iz == 0);

  } // end for ivar

} // FillBlockGhostCellsFunctor<dim, device_t>::conservative_interpolation_order4 - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::fill_inner(int32_t cellindex_in,
                                                      int32_t cellindex_out,
                                                      int32_t iOct_global) const
{

  const auto nbvar = m_userdata_in.num_vars();

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(cellindex_in, ivar, iOct_global);

} // FillBlockGhostCellsFunctor<dim, device_t>::fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
                                                       coord_t<dim> const & coord_out,
                                                       int32_t const &      iOct_global) const
{

  const auto & b = m_userdata_in.block_size();

  coord_t<dim> coord_in;
  const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

  int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_userdata_in.block_size());

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  if (dir_norm == 0)
  {
    // current cell is inside current block
    fill_inner(cellindex_in, cellindex_out, iOct_global);
  }
  else
  {
    // current cell is a ghost cell (thus belonging to a neighbor block)

    /*
     * fill ghosts all around
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_global);

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const CellLocation_t cell_loc_cur{ coord_in, key_cur, iOct_global, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    const auto nbvar = m_userdata_in.num_vars();

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
      cellindex_in = cell_loc_neigh.cellindex(m_userdata_in.block_size());

      for (int32_t ivar = 0; ivar < nbvar; ++ivar)
      {
        m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
          m_userdata_in(cellindex_in, ivar, iOct_in);
      }
    }
    else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser

      if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value

        cellindex_in = cell_loc_neigh.cellindex(m_userdata_in.block_size());

        for (int32_t ivar = 0; ivar < nbvar; ++ivar)
        {
          m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
            m_userdata_in(cellindex_in, ivar, iOct_in);
        }
      }
      else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        linear_extrapolate_using_limited_slopes(
          cell_loc_neigh, coord_in, cellindex_out, iOct_global);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_2)
      {
        conservative_interpolation_order2(cell_loc_neigh, coord_out, cellindex_out, iOct_global);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_4)
      {
        conservative_interpolation_order4(cell_loc_neigh, coord_out, cellindex_out, iOct_global);
      }
      else
      {
        // we shouldn't be here
        KOKKOS_ASSERT(false && "Unsupported prolongation type");
      }
    }
    else if (cell_loc_neigh.level() == cell_loc_cur.level() + 1)
    {
      // doing a restriction
      for (int32_t ivar = 0; ivar < nbvar; ++ivar)
      {
        m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
          m_stencil_helper.compute_siblings_average(
            cell_loc_neigh, m_userdata_in.block_size(), ivar, m_userdata_in);

      } // for ivar
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm == 0)

} // FillBlockGhostCellsFunctor<dim, device_t>::fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  const auto iOct_local = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct_local * nbCellsPerGhostedLeaf;

  const auto iOct_global = m_iOct_begin + iOct_local;

  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global);

} // FillBlockGhostCellsFunctor<dim, device_t>::operator()

template class FillBlockGhostCellsFunctor<2, kalypsso::DefaultDevice>;
template class FillBlockGhostCellsFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
