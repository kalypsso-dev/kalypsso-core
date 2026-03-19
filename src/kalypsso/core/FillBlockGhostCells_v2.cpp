// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostCells_v2.cpp
 * \brief \copybrief FillBlockGhostCells_v2.h
 */

#include <kalypsso/core/FillBlockGhostCells_v2.h>

namespace kalypsso
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
FillBlockGhostCellsFunctorV2<dim, device_t>::FillBlockGhostCellsFunctorV2(
  amr_hashmap_t            amr_hashmap,
  orchard_key_view_t       orchard_keys,
  int32_t                  iOct_begin,
  int32_t                  num_octants,
  DataArrayBlock_t         userdata_in,
  DataArrayGhostedBlock_t  userdata_out,
  brick_size_t<dim>        brick_sizes,
  Kokkos::Array<bool, dim> is_brick_periodic)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_userdata_in(userdata_in)
  , m_userdata_out(userdata_out)
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
{
  KOKKOS_ASSERT(userdata_out.block_size() == userdata_in.block_size() &&
                "userdata_in and userdata_out must have the same block sizes.");
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillBlockGhostCellsFunctorV2<dim, device_t>::apply(amr_hashmap_t            amr_hashmap,
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

  FillBlockGhostCellsFunctorV2<dim, device_t> functor(amr_hashmap,
                                                      orchard_keys,
                                                      iOct_begin,
                                                      num_octants,
                                                      userdata_in,
                                                      userdata_out,
                                                      brick_sizes,
                                                      is_brick_periodic);

  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_octants * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for(
    "FillBlockGhostCellsFunctorV2", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor);

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
FillBlockGhostCellsFunctorV2<dim, device_t>::compute_siblings_average(coord_t<dim> const & coords,
                                                                      int32_t const &      ivar,
                                                                      int32_t const & iOct) const
{
  const auto & b = m_userdata_in.block_size();

  // number of siblings :
  // - 4 in 2d
  // - 8 in 3d
  // that is 2^dim
  constexpr int num_siblings = 1 << dim;

  // average child octant values
  real_t value = 0;

  if constexpr (dim == 2)
  {
    for (int32_t jj = 0; jj < 2; ++jj)
    {
      for (int32_t ii = 0; ii < 2; ++ii)
      {

        KOKKOS_ASSERT((coords[IX] + ii < b[IX]) && "Wrong values for coords[IX]");
        KOKKOS_ASSERT((coords[IY] + jj < b[IY]) && "Wrong values for coords[IY]");

        // compute corresponding index in the ghosted block (current octant)
        const auto coord_sibling = coord_t<2>{ coords[IX] + ii, coords[IY] + jj };
        const auto cellindex_sibling = coord_to_cellindex<dim>(coord_sibling, b);

        value += m_userdata_in(cellindex_sibling, ivar, iOct);
      } // for ii
    } // for jj
  }
  else if constexpr (dim == 3)
  {
    for (int32_t kk = 0; kk < 2; ++kk)
    {
      for (int32_t jj = 0; jj < 2; ++jj)
      {
        for (int32_t ii = 0; ii < 2; ++ii)
        {

          KOKKOS_ASSERT((coords[IX] + ii < b[IX]) && "Wrong values for coords[IX]");
          KOKKOS_ASSERT((coords[IY] + jj < b[IY]) && "Wrong values for coords[IY]");
          KOKKOS_ASSERT((coords[IZ] + kk < b[IZ]) && "Wrong values for coords[IZ]");

          // compute corresponding index in the ghosted block (current octant)
          const auto coord_sibling =
            coord_t<3>{ coords[IX] + ii, coords[IY] + jj, coords[IZ] + kk };
          const auto cellindex_sibling = coord_to_cellindex<dim>(coord_sibling, b);

          value += m_userdata_in(cellindex_sibling, ivar, iOct);
        } // for ii
      } // for jj
    } // for kk
  }

  return value / num_siblings;
} // compute_siblings_average

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctorV2<dim, device_t>::fill_inner(int32_t cellindex_in,
                                                        int32_t cellindex_out,
                                                        int32_t iOct_global) const
{
  const auto nbvar = m_userdata_in.num_vars();

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(cellindex_in, ivar, iOct_global);

} // fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctorV2<dim, device_t>::fill_ghost_same_level(int32_t iOct_global,
                                                                   int32_t iOct_neigh,
                                                                   index_t cellindex_in,
                                                                   index_t cellindex_out) const
{

  const auto nbvar = m_userdata_in.num_vars();

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(cellindex_in, ivar, iOct_neigh);

} // fill_ghost_same_level

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctorV2<dim, device_t>::fill_ghost_coarser_level(uint8_t      child_id,
                                                                      int32_t      iOct_global,
                                                                      int32_t      iOct_neigh,
                                                                      coord_t<dim> coord_in,
                                                                      index_t cellindex_out) const
{

  const auto & b = m_userdata_in.block_size();

  coord_in[IX] = (coord_in[IX] + ((child_id & 0x1) >> 0) * b[IX]) / 2;
  coord_in[IY] = (coord_in[IY] + ((child_id & 0x2) >> 1) * b[IY]) / 2;
  if constexpr (dim == 3)
  {
    coord_in[IZ] = (coord_in[IZ] + ((child_id & 0x4) >> 2) * b[IZ]) / 2;
  }

  auto cellindex_neigh = coord_to_cellindex<dim>(coord_in, m_userdata_in.block_size());

  KOKKOS_ASSERT((coord_in[IX] < b[IX]) && "Wrong values for coord_in[IX]");
  KOKKOS_ASSERT((coord_in[IY] < b[IY]) && "Wrong values for coord_in[IY]");
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT((coord_in[IZ] < b[IZ]) && "Wrong values for coord_in[IZ]");
  }

  const auto nbvar = m_userdata_in.num_vars();

  for (int32_t ivar = 0; ivar < nbvar; ++ivar)
    m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
      m_userdata_in(cellindex_neigh, ivar, iOct_neigh);

} // fill_ghost_coarser_level

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION bool
FillBlockGhostCellsFunctorV2<dim, device_t>::fill_ghost_finer_level(key_t   key_neigh_same_level,
                                                                    int32_t iOct_global,
                                                                    coord_t<dim> coord_in,
                                                                    index_t cellindex_out) const
{
  auto status = false;

  const auto & b = m_userdata_in.block_size();

  coord_in = coord_in * 2;

  // determine in which finer octant we need to use (i.e. determine child id)
  uint8_t child_id = 0;
  if (coord_in[IX] >= b[IX])
  {
    coord_in[IX] -= b[IX];
    child_id += 1;
  }
  if (coord_in[IY] >= b[IY])
  {
    coord_in[IY] -= b[IY];
    child_id += 2;
  }
  if constexpr (dim == 3)
  {
    if (coord_in[IZ] >= b[IZ])
    {
      coord_in[IZ] -= b[IZ];
      child_id += 4;
    }
  }

  // now coord_in contain the coordinates of the lower left finer cell to use
  // we need to average this cell with its siblings

  // lookup for this child in the hashmap
  const auto key_neigh_fine = orchard_key_t<dim>::child(key_neigh_same_level, child_id);

  const auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_fine);
  const auto valid = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

  if (valid)
  {

    const auto iOct_neigh =
      static_cast<int32_t>(m_amr_hashmap_device.value_at(key_neigh_hashindex));

    // fill ghost cell with values at finer level

    const auto nbvar = m_userdata_in.num_vars();

    for (int32_t ivar = 0; ivar < nbvar; ++ivar)
    {
      m_userdata_out(cellindex_out, ivar, iOct_global - m_iOct_begin) =
        compute_siblings_average(coord_in, ivar, iOct_neigh);

    } // for ivar

    status = true;
  }
  else
  {
    status = false;
  }

  return status;
} // fill_ghost_finer_level

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctorV2<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
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
    /*
     * fill ghosts
     */

    // get orchard key of current octant
    auto key_cur = m_orchard_keys_device(iOct_global);

    // get neighbor key at same level
    auto key_neigh_same_level = orchard_key_t<dim>::get_neighbor_key_same_level(
      key_cur, dir, m_brick_sizes, m_is_brick_periodic);

    // check if neighbor key exists in hash map
    auto key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_same_level);

    auto is_at_same_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

    // if key is valid, it means neighbor is actually at the same level
    if (is_at_same_level)
    {
      const auto iOct_neigh =
        static_cast<int32_t>(m_amr_hashmap_device.value_at(key_neigh_hashindex));

      fill_ghost_same_level(iOct_global, iOct_neigh, cellindex_in, cellindex_out);
    }
    else
    {

      // check if father exist (coarser level)
      auto key_neigh_coarser = orchard_key_t<dim>::father(key_neigh_same_level);

      key_neigh_hashindex = m_amr_hashmap_device.find(key_neigh_coarser);

      auto is_at_coarser_level = m_amr_hashmap_device.valid_at(key_neigh_hashindex);

      if (is_at_coarser_level)
      {
        const auto iOct_neigh =
          static_cast<int32_t>(m_amr_hashmap_device.value_at(key_neigh_hashindex));
        const auto child_id = orchard_key_t<dim>::child_id(key_neigh_same_level);

        fill_ghost_coarser_level(child_id, iOct_global, iOct_neigh, coord_in, cellindex_out);
      }
      else
      {
        // try to fill ghost cell with data at finer level
        auto fill_ok =
          fill_ghost_finer_level(key_neigh_same_level, iOct_global, coord_in, cellindex_out);

        // if not successful, it means neighbor octant doesn't exist in hashmap
        // this can only happen if the ghost cell is outside domain border and we are not using
        // periodic brick connectivity
        if (!fill_ok)
        {
          if (orchard_key_t<dim>::is_at_any_domain_border(key_cur, m_brick_sizes))
          {
            orchard_key_t<dim>::print(key_cur);
            // TODO: provides default ways to fill ghost cells
            // or let the numerical scheme decide
          }
          else
          {
            KOKKOS_ASSERT(false && "Logic error: neighbor not found");
          }
        }
      }
    }
  }

} // fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostCellsFunctorV2<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  const auto iOct_local = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct_local * nbCellsPerGhostedLeaf;

  const auto iOct_global = m_iOct_begin + iOct_local;
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global);

} // operator()

template class FillBlockGhostCellsFunctorV2<2, kalypsso::DefaultDevice>;
template class FillBlockGhostCellsFunctorV2<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
