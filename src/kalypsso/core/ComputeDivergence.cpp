// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDivergence.cpp
 */
#include <kalypsso/core/ComputeDivergence.h>

namespace kalypsso
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
ComputeDivergence<dim, device_t>::run(amr_hashmap_t            amr_hashmap,
                                      orchard_key_view_t       orchard_keys,
                                      int32_t                  local_num_octants,
                                      block_size_t<dim>        block_sizes,
                                      brick_size_t<dim>        brick_sizes,
                                      Kokkos::Array<bool, dim> is_brick_periodic,
                                      real_t                   scaling_factor,
                                      DataArrayBlock_t         userdata,
                                      Kokkos::Array<int, dim>  field_index) -> DataArrayBlock_t
{
  auto divergence = DataArrayBlock_t("DivergenceB", userdata.block_size(), 1, local_num_octants);

  ComputeDivergence<dim, device_t> functor(amr_hashmap,
                                           orchard_keys,
                                           local_num_octants,
                                           block_sizes,
                                           brick_sizes,
                                           is_brick_periodic,
                                           scaling_factor,
                                           userdata,
                                           divergence,
                                           field_index);

  const auto nbCellsPerLeaf = Kokkos::dim_prod(block_sizes);
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for(
    "compute_divergence_B", Kokkos::RangePolicy<exec_space>(0, nbCellsTotal), functor);

  return divergence;

} // ComputeDivergence<dim, device_t>::run

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeDivergence<dim, device_t>::getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                                                           int32_t                   cell_index,
                                                           shift_t<dim>              shift,
                                                           int const &               ivar) const
{
  const auto cell_loc_neigh = m_helper.getNeighLoc(cell_loc, shift);
  const auto cell_index_neigh = coord_to_cellindex<dim>(cell_loc_neigh.ijk, m_block_sizes);

  if (cell_loc_neigh.is_outside_domain)
  {
    // return value in current cell
    return m_userdata_in(cell_index, ivar, cell_loc.iOct);
  }
  else
  {
    // if neighbor is at the same level or is coarser, juste use the value
    if ((cell_loc_neigh.level() == cell_loc.level()) or
        (cell_loc_neigh.level() == cell_loc.level() - 1))
    {
      KOKKOS_ASSERT(static_cast<int32_t>(cell_loc_neigh.iOct) < m_userdata_in.num_quadrants() &&
                    "userdata has wrong size. You probability forgot to update/resize it.");
      return m_userdata_in(cell_index_neigh, ivar, cell_loc_neigh.iOct);
    }
    // if neighbor is finer, average small cell values
    else if (cell_loc_neigh.level() == cell_loc.level() + 1)
    {
      return m_helper.compute_siblings_average(cell_loc_neigh, m_block_sizes, ivar, m_userdata_in);
    }
    else
    {
      // we shouldn't be here
      return ZERO_F;
    }
  }

} // ComputeDivergence<dim, device_t>::getNeighborDataSameLevel

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeDivergence<dim, device_t>::compute_divergence(CellLocation_t const & cell_loc,
                                                     int32_t const &        cell_index) const
{

  auto data_L =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IX>(), m_field_index[IX]);

  auto data_R =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IX>(), m_field_index[IX]);

  auto div = (data_R - data_L);


  data_L =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IY>(), m_field_index[IY]);

  data_R =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IY>(), m_field_index[IY]);

  div += (data_R - data_L);

  if constexpr (dim == 3)
  {
    data_L =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IZ>(), m_field_index[IZ]);

    data_R =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IZ>(), m_field_index[IZ]);

    div += (data_R - data_L);
  }

  return div;

} // ComputeDivergence<dim, device_t>::compute_divergence

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeDivergence<dim, device_t>::operator()(const index_t & global_index) const
{
  const auto iOct_global = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct_global * m_nbCellsPerLeaf;

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_block_sizes);
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  auto const level = orchard_key_t<dim>::level(m_orchard_keys_device(iOct_global));
  auto const dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;


  auto div = compute_divergence(cell_loc, cell_index) / dx;

  m_userdata_out(cell_index, 0, iOct_global) = div;

} // ComputeDivergence<dim, device_t>::operator()

template class ComputeDivergence<2, kalypsso::DefaultDevice>;
template class ComputeDivergence<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
