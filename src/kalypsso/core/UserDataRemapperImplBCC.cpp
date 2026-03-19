// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplBCC.cpp
 * \brief \copybrief UserDataRemapperImplBCC.h
 */
#include <kalypsso/core/UserDataRemapperImplBCC.h>

namespace kalypsso
{
// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::operator()(const index_t & global_index) const
{
  // get (octant id, cellindex) from global_index
  // using left indexing, we get
  // global_index = cellindex + iOct * num_cells
  const auto num_cells = m_userdataBlock_old.num_cells();

  const auto iOct = global_index / num_cells;
  const auto cellindex = global_index - iOct * num_cells;

  const auto num_vars = m_userdataBlock_old.num_vars();
  const auto bSize = m_userdataBlock_old.block_size();

  // get orchard key of current block/octant
  const auto key_new = m_orchard_keys_device_new(iOct);

  // for each key try to find it in the old hashmap
  auto key_index = m_amr_hashmap_device_old.find(key_new);

  // first check if key exists in the old hashmap, if it exists, it means we have a
  // an octant that didn't change level
  if (m_amr_hashmap_device_old.valid_at(key_index))
  {
    // auto key_old = amr_hashmap_device_old.key_at(key_index);
    auto iOct_old = m_amr_hashmap_device_old.value_at(key_index);

    for (int32_t ivar = 0; ivar < num_vars; ++ivar)
    {
      // copy from old quadrant to new quadrant
      m_userdataBlock_new(cellindex, ivar, iOct) = m_userdataBlock_old(cellindex, ivar, iOct_old);
    }
  }
  else
  {
    // check if the key correspond to a refinement (increase of level / prolongation)
    // to do that we search for father octant's key
    auto key_to_test = orchard_key_t<dim>::father(key_new);
    key_index = m_amr_hashmap_device_old.find(key_to_test);

    if (m_amr_hashmap_device_old.valid_at(key_index))
    {
      auto iOct_old = m_amr_hashmap_device_old.value_at(key_index);

      // get family id of the new key, used to determine where exactly in the old block we
      // will copy data
      auto fid = static_cast<int32_t>(orchard_key_t<dim>::family_id(key_new));

      // transform iCoord (new cell) into iCoord (corresponding coarse cell in old
      // quadrant)
      auto iCoord = cellindex_to_coord<dim>(cellindex, bSize);

      coord_t<dim> iCoord_old;
      if constexpr (dim == 2)
      {
        iCoord_old[IX] = iCoord[IX] / 2 + ((fid & 0x1) >> 0) * (bSize[IX] / 2);
        iCoord_old[IY] = iCoord[IY] / 2 + ((fid & 0x2) >> 1) * (bSize[IY] / 2);
      }
      else
      {
        iCoord_old[IX] = iCoord[IX] / 2 + ((fid & 0x1) >> 0) * (bSize[IX] / 2);
        iCoord_old[IY] = iCoord[IY] / 2 + ((fid & 0x2) >> 1) * (bSize[IY] / 2);
        iCoord_old[IZ] = iCoord[IZ] / 2 + ((fid & 0x4) >> 2) * (bSize[IZ] / 2);
      }
      const auto cellindex_old = coord_to_cellindex<dim>(iCoord_old, bSize);

      if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value from the old mesh
        for (int32_t ivar = 0; ivar < num_vars; ++ivar)
        {
          m_userdataBlock_new(cellindex, ivar, iOct) =
            m_userdataBlock_old(cellindex_old, ivar, iOct_old);
        } // end for ivar
      }
      else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        // current cell location in old mesh
        const CellLocation_t cell_loc_old{ iCoord_old, key_to_test, iOct_old, false };
        linear_extrapolate_using_limited_slopes(cell_loc_old, iCoord, cellindex, iOct);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_2)
      {
        // current cell location in old mesh
        const CellLocation_t cell_loc_old{ iCoord_old, key_to_test, iOct_old, false };
        conservative_interpolation_order2(cell_loc_old, iCoord, cellindex, iOct);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_4)
      {
        // current cell location in old mesh
        const CellLocation_t cell_loc_old{ iCoord_old, key_to_test, iOct_old, false };
        conservative_interpolation_order4(cell_loc_old, iCoord, cellindex, iOct);
      }
      else
      {
        // Houston we have a problem....
        // we shouldn't be here
        KOKKOS_ASSERT(false && "Unsupported prolongation type");
      }
    }
    else
    {
      // check if the key correspond to a "coarsening" (decrease of level / restriction)
      // to do that we just need to search for the eldest child key in the old map,
      // and then average all the values from all the children (coarse graining values)

      /*
       * In principe, we should check that all the children are available in the map
       * but here, we just check for the eldest child key, and assume all the children
       * are also in the map (p4est ensure that anyway).
       */

      constexpr auto NB_CHILDREN = orchard_key_t<dim>::NB_CHILDREN;

      auto eldest_child_key = orchard_key_t<dim>::eldest_child(key_new);
      key_index = m_amr_hashmap_device_old.find(eldest_child_key);
      if (m_amr_hashmap_device_old.valid_at(key_index))
      {
        auto iOct_eldest = m_amr_hashmap_device_old.value_at(key_index);

        for (int32_t ivar = 0; ivar < num_vars; ++ivar)
        {
          // transform iCoord (new cell) into iCoord (corresponding coarse cell in old
          // quadrant)
          auto iCoord = cellindex_to_coord<dim>(cellindex, bSize);

          // for a given cell determine the child id where we will need to fetch data in the
          // old mesh
          uint32_t ichild = 0;
          if (iCoord[IX] >= bSize[IX] / 2)
            ichild |= 0x1;
          if (iCoord[IY] >= bSize[IY] / 2)
            ichild |= 0x2;
          if constexpr (dim == 3)
          {
            if (iCoord[IZ] >= bSize[IZ] / 2)
            {
              ichild |= 0x4;
            }
          }

          // here we taken into account that a family of octant are stored contiguously in
          // memory
          auto iOct_old = iOct_eldest + ichild;

          // we need to average value over NB_CHILDREN neighboring cells
          real_t coarsen_value = ZERO_F;
          if constexpr (dim == 2)
          {
            for (uint8_t iy = 0; iy < 2; ++iy)
            {
              auto iiy = 2 * iCoord[IY] + iy >= bSize[IY] ? 2 * iCoord[IY] + iy - bSize[IY]
                                                          : 2 * iCoord[IY] + iy;

              for (uint8_t ix = 0; ix < 2; ++ix)
              {
                auto iix = 2 * iCoord[IX] + ix >= bSize[IX] ? 2 * iCoord[IX] + ix - bSize[IX]
                                                            : 2 * iCoord[IX] + ix;
                coarsen_value += m_userdataBlock_old(iix, iiy, ivar, iOct_old);
              }
            }
          }
          else
          {
            for (uint8_t iz = 0; iz < 2; ++iz)
            {
              auto iiz = 2 * iCoord[IZ] + iz >= bSize[IZ] ? 2 * iCoord[IZ] + iz - bSize[IZ]
                                                          : 2 * iCoord[IZ] + iz;
              for (uint8_t iy = 0; iy < 2; ++iy)
              {
                auto iiy = 2 * iCoord[IY] + iy >= bSize[IY] ? 2 * iCoord[IY] + iy - bSize[IY]
                                                            : 2 * iCoord[IY] + iy;
                for (uint8_t ix = 0; ix < 2; ++ix)
                {
                  auto iix = 2 * iCoord[IX] + ix >= bSize[IX] ? 2 * iCoord[IX] + ix - bSize[IX]
                                                              : 2 * iCoord[IX] + ix;
                  coarsen_value += m_userdataBlock_old(iix, iiy, iiz, ivar, iOct_old);
                }
              }
            }
          }

          m_userdataBlock_new(cellindex, ivar, iOct) = coarsen_value / NB_CHILDREN;
        }

        // average value since we are coarsening
        // WARNING : here we assume all the children are available in the map
        // if this is not true, it means the map is corrupted => should abort
        // in principle, this is highly not probable because p4est already checked that
        // the whole family of octant are present
        // for (size_t ichild = 0; ichild < NB_CHILDREN; ++ichild)
        // {
        //   value += m_userdataLeaf(index_old + ichild, 0);
        // }
      }
    }
  }

} // operator()

// ==============================================================
// ==============================================================
/**
 * Do linear extrapolation using limited slopes.
 */
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const & cell_loc_old,
  coord_t<2> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const real_t slope_type = 1;
  const auto   nbvar = m_userdataBlock_old.num_vars();

  // get neighbor location, each of then may be at finer, same or coarser level compared
  // cell_loc_old
  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current new cell inside old parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(
      cell_loc_old, cell_loc_right_x, cell_loc_left_x, ivar, m_userdataBlock_old, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(
      cell_loc_old, cell_loc_right_y, cell_loc_left_y, ivar, m_userdataBlock_old, slope_type);

    // extrapolate
    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_userdataBlock_old(cell_loc_old.cellindex(m_block_sizes), ivar, cell_loc_old.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx + ONE_FOURTH_F * static_cast<real_t>(iy) * dudy;
  }
} // linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
/**
 * Do linear extrapolation using limited slopes.
 */
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const & cell_loc_old,
  coord_t<3> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const real_t slope_type = 1;
  const auto   nbvar = m_userdataBlock_old.num_vars();

  // get neighbor location, each of then may be at finer, same or coarser level compared
  // cell_loc_old
  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current new cell inside old parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;
  const int iz = 2 * (coord_new[IZ] - 2 * (coord_new[IZ] / 2)) - 1;

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    // compute limited slopes
    auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(
      cell_loc_old, cell_loc_right_x, cell_loc_left_x, ivar, m_userdataBlock_old, slope_type);
    auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(
      cell_loc_old, cell_loc_right_y, cell_loc_left_y, ivar, m_userdataBlock_old, slope_type);
    auto const dudz = m_stencil_helper.compute_minmod_slopes_prolongation(
      cell_loc_old, cell_loc_right_z, cell_loc_left_z, ivar, m_userdataBlock_old, slope_type);

    // extrapolate
    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_userdataBlock_old(cell_loc_old.cellindex(m_block_sizes), ivar, cell_loc_old.iOct) +
      ONE_FOURTH_F * static_cast<real_t>(ix) * dudx +
      ONE_FOURTH_F * static_cast<real_t>(iy) * dudy + ONE_FOURTH_F * static_cast<real_t>(iz) * dudz;
  }
} // linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order2(
  CellLocation<2> const & cell_loc_old,
  coord_t<2> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdataBlock_old.num_vars();

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
  const int ix = (coord_new[IX] - 2 * (coord_new[IX] / 2));
  const int iy = (coord_new[IY] - 2 * (coord_new[IY] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t val[3];

    for (int j = -1; j < 2; ++j)
    {
      // interpolate along X
      auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<2>{ -1, j });
      auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<2>{ 0, j });
      auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<2>{ 1, j });

      val[j + 1] = m_stencil_helper.compute_linear_combination(cell_loc_x_m1,
                                                               cell_loc_x_0,
                                                               cell_loc_x_p1,
                                                               cell_loc_old.level(),
                                                               ivar,
                                                               m_userdataBlock_old,
                                                               ix == 0 ? m_cons_interpol.COEFS2_L
                                                                       : m_cons_interpol.COEFS2_R);
    }

    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_cons_interpol.order2(val[0], val[1], val[2], iy == 0);

  } // end for ivar

} // UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order2 - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order2(
  CellLocation<3> const & cell_loc_old,
  coord_t<3> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdataBlock_old.num_vars();

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
  const int ix = (coord_new[IX] - 2 * (coord_new[IX] / 2));
  const int iy = (coord_new[IY] - 2 * (coord_new[IY] / 2));
  const int iz = (coord_new[IZ] - 2 * (coord_new[IZ] / 2));

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
  {
    real_t valx[3][3];

    for (int k = -1; k < 2; ++k)
    {
      for (int j = -1; j < 2; ++j)
      {
        // interpolate along X
        auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<3>{ -1, j, k });
        auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<3>{ 0, j, k });
        auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_old, shift_t<3>{ 1, j, k });

        valx[k + 1][j + 1] = m_stencil_helper.compute_linear_combination(
          cell_loc_x_m1,
          cell_loc_x_0,
          cell_loc_x_p1,
          cell_loc_old.level(),
          ivar,
          m_userdataBlock_old,
          ix == 0 ? m_cons_interpol.COEFS2_L : m_cons_interpol.COEFS2_R);
      }
    }
    real_t valy[3];

    valy[0] = m_cons_interpol.order2(valx[0][0], valx[0][1], valx[0][2], iy == 0);
    valy[1] = m_cons_interpol.order2(valx[1][0], valx[1][1], valx[1][2], iy == 0);
    valy[2] = m_cons_interpol.order2(valx[2][0], valx[2][1], valx[2][2], iy == 0);

    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_cons_interpol.order2(valy[0], valy[1], valy[2], iz == 0);

  } // end for ivar

} // UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order2 - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order4(
  CellLocation<2> const & cell_loc_neigh,
  coord_t<2> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdataBlock_old.num_vars();

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
  const int ix = (coord_new[IX] - 2 * (coord_new[IX] / 2));
  const int iy = (coord_new[IY] - 2 * (coord_new[IY] / 2));

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
                                                               m_userdataBlock_old,
                                                               ix == 0 ? m_cons_interpol.COEFS4_L
                                                                       : m_cons_interpol.COEFS4_R);
    }

    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_cons_interpol.order4(val[0], val[1], val[2], val[3], val[4], iy == 0);

  } // end for ivar

} // UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order4 - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order4(
  CellLocation<3> const & cell_loc_neigh,
  coord_t<3> const &      coord_new,
  index_t const &         cellindex_new,
  int32_t const &         iOct_global) const
{
  const auto nbvar = m_userdataBlock_old.num_vars();

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
  const int ix = (coord_new[IX] - 2 * (coord_new[IX] / 2));
  const int iy = (coord_new[IY] - 2 * (coord_new[IY] / 2));
  const int iz = (coord_new[IZ] - 2 * (coord_new[IZ] / 2));

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
          m_userdataBlock_old,
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

    m_userdataBlock_new(cellindex_new, ivar, iOct_global) =
      m_cons_interpol.order4(valy[0], valy[1], valy[2], valy[3], valy[4], iz == 0);

  } // end for ivar

} // UserDataRemapperImplBCC<dim, device_t>::conservative_interpolation_order4 - 3d


// explicit template instantiation
template class UserDataRemapperImplBCC<2, kalypsso::DefaultDevice>;
template class UserDataRemapperImplBCC<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class UserDataRemapperImplBCC<2, HostDevice>;
template class UserDataRemapperImplBCC<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso
