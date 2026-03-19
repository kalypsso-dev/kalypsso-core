// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeRefineFlags.cpp
 */
#include <kalypsso/core/ComputeRefineFlags.h>

namespace kalypsso
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeRefineFlags<dim, device_t>::run(amr_hashmap_t            amr_hashmap,
                                       orchard_key_view_t       orchard_keys,
                                       int32_t                  local_num_octants,
                                       brick_size_t<dim>        brick_sizes,
                                       Kokkos::Array<bool, dim> is_brick_periodic,
                                       DataArrayBlock_t         userdata,
                                       amrflags_view_t          flags,
                                       RefineIndicatorData      refineParams)
{

  ComputeRefineFlags<dim, device_t> functor(amr_hashmap,
                                            orchard_keys,
                                            local_num_octants,
                                            brick_sizes,
                                            is_brick_periodic,
                                            userdata,
                                            flags,
                                            refineParams);

  const auto nbCellsPerLeaf = userdata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  if (refineParams.indicator == +Indicator::LOHNER_SPLIT)
  {
    Kokkos::parallel_for("ComputeRefineFlags",
                         Kokkos::RangePolicy<exec_space, TagLohnerSplit>(0, nbCellsTotal),
                         functor);
  }
  else if (refineParams.indicator == +Indicator::LOHNER_UNSPLIT)
  {
    Kokkos::parallel_for("ComputeRefineFlags",
                         Kokkos::RangePolicy<exec_space, TagLohnerUnsplit>(0, nbCellsTotal),
                         functor);
  }
  else if (refineParams.indicator == +Indicator::SIMPLE_GRADIENT)
  {
    Kokkos::parallel_for("ComputeRefineFlags",
                         Kokkos::RangePolicy<exec_space, TagSimpleGradient>(0, nbCellsTotal),
                         functor);
  }
  else if (refineParams.indicator == +Indicator::THRESHOLD_AFFINE)
  {
    Kokkos::parallel_for("ComputeRefineFlags",
                         Kokkos::RangePolicy<exec_space, TagThresholdAffine>(0, nbCellsTotal),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

}; // ComputeRefineFlags<dim, device_t>::run

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                                                            int32_t                   cell_index,
                                                            shift_t<dim>              shift,
                                                            int                       ivar) const
{

  const auto cell_loc_neigh = m_helper.getNeighLoc(cell_loc, shift);
  const auto cell_index_neigh =
    coord_to_cellindex<dim>(cell_loc_neigh.ijk, m_userdata.block_size());

  if (cell_loc_neigh.is_outside_domain)
  {
    // return value in current cell
    return m_userdata(cell_index, ivar, cell_loc.iOct);
  }
  else
  {
    // if neighbor is at same level or is coarser, just use the value
    if ((cell_loc_neigh.level() == cell_loc.level()) or
        (cell_loc_neigh.level() == cell_loc.level() - 1))
    {
      KOKKOS_ASSERT(static_cast<int32_t>(cell_loc_neigh.iOct) < m_userdata.num_quadrants() &&
                    "userdata has wrong size. You probability forgot to update/resize it.");
      return m_userdata(cell_index_neigh, ivar, cell_loc_neigh.iOct);
    }
    // if neighbor is finer, average small cell values
    else if (cell_loc_neigh.level() == cell_loc.level() + 1)
    {
      return m_helper.compute_siblings_average(
        cell_loc_neigh, m_userdata.block_size(), ivar, m_userdata);
    }
    else
    {
      // we shouldn't be here
      return ZERO_F;
    }
  }

} // ComputeRefineFlags<dim, device_t>::getNeighborDataSameLevel

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::normalized_gradient(real_t const & v1, real_t const & v2) const
{

  auto vmax = fmax(fabs(v1), fabs(v2));

  if (vmax < KALYPSSO_NUM(0.001))
  {
    return ZERO_F;
  }

  vmax = fabs(v1 - v2) / vmax;
  return fmax(fmin(vmax, ONE_F), ZERO_F);

} // ComputeRefineFlags<dim, device_t>::normalized_gradient

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <int dir>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::compute_lohner_split(CellLocation_t const & cell_loc,
                                                        int32_t const &        cell_index,
                                                        int const &            ivar) const
{

  constexpr auto shift_L = get_shift_left<dim, dir>();
  constexpr auto shift_R = get_shift_right<dim, dir>();

  const auto data_L = getNeighborDataSameLevel(cell_loc, cell_index, shift_L, ivar);
  const auto data_R = getNeighborDataSameLevel(cell_loc, cell_index, shift_R, ivar);
  const auto data_C = m_userdata(cell_index, ivar, cell_loc.iOct);

  return abs(data_L - 2 * data_C + data_R) /
         (abs(data_R - data_C) + abs(data_C - data_L) +
          m_refineParams.epsilon * (abs(data_L) + 2 * abs(data_C) + abs(data_R)));

} // ComputeRefineFlags<dim, device_t>::compute_lohner_split

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::compute_lohner_unsplit(CellLocation_t const & cell_loc,
                                                          int32_t const &        cell_index,
                                                          int const &            ivar) const
{

  const auto data_C = m_userdata(cell_index, ivar, cell_loc.iOct);

  const auto data_Lx =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IX>(), ivar);
  const auto data_Rx =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IX>(), ivar);

  const auto data_Ly =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IY>(), ivar);
  const auto data_Ry =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IY>(), ivar);

  auto num = (data_Lx - 2 * data_C + data_Rx) * (data_Lx - 2 * data_C + data_Rx) +
             (data_Ly - 2 * data_C + data_Ry) * (data_Ly - 2 * data_C + data_Ry);

  auto tmp = abs(data_Rx - data_C) + abs(data_C - data_Lx) +
             m_refineParams.epsilon * (abs(data_Lx) + 2 * abs(data_C) + abs(data_Rx));
  auto denom = tmp * tmp;

  tmp = abs(data_Ry - data_C) + abs(data_C - data_Ly) +
        m_refineParams.epsilon * (abs(data_Ly) + 2 * abs(data_C) + abs(data_Ry));
  denom += tmp * tmp;


  if constexpr (dim == 3)
  {
    const auto data_Lz =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IZ>(), ivar);
    const auto data_Rz =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IZ>(), ivar);

    num += (data_Lz - 2 * data_C + data_Rz) * (data_Lz - 2 * data_C + data_Rz);

    tmp = abs(data_Rz - data_C) + abs(data_C - data_Lz) +
          m_refineParams.epsilon * (abs(data_Lz) + 2 * abs(data_C) + abs(data_Rz));
    denom += tmp * tmp;
  }

  return sqrt(num / denom);

} // ComputeRefineFlags<dim, device_t>::compute_lohner_unsplit

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::compute_simple_gradient(CellLocation_t const & cell_loc,
                                                           int32_t const &        cell_index,
                                                           int const &            ivar) const
{

  auto res = Kokkos::reduction_identity<real_t>::max();

  const auto data_C = m_userdata(cell_index, ivar, cell_loc.iOct);

  const auto data_Lx =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IX>(), ivar);
  const auto data_Rx =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IX>(), ivar);

  res = fmax(res, normalized_gradient(data_C, data_Lx));
  res = fmax(res, normalized_gradient(data_C, data_Rx));

  const auto data_Ly =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IY>(), ivar);
  const auto data_Ry =
    getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IY>(), ivar);

  res = fmax(res, normalized_gradient(data_C, data_Ly));
  res = fmax(res, normalized_gradient(data_C, data_Ry));

  if constexpr (dim == 3)
  {
    const auto data_Lz =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_left<dim, IZ>(), ivar);
    const auto data_Rz =
      getNeighborDataSameLevel(cell_loc, cell_index, get_shift_right<dim, IZ>(), ivar);

    res = fmax(res, normalized_gradient(data_C, data_Lz));
    res = fmax(res, normalized_gradient(data_C, data_Rz));
  }

  return res;

} // ComputeRefineFlags<dim, device_t>::compute_simple_gradient

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeRefineFlags<dim, device_t>::compute_threshold_affine(
  CellLocation_t const &        cell_loc,
  int32_t const &               cell_index,
  int const &                   ivar,
  ThresholdAffineParams const & th_aff_params) const
{

  const auto data_C = m_userdata(cell_index, ivar, cell_loc.iOct);

  return th_aff_params.a * data_C + th_aff_params.b;

} // ComputeRefineFlags<dim, device_t>::compute_threshold_affine

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeRefineFlags<dim, device_t>::update_flag(amrflag_t &     flag,
                                               real_t const &  indicator,
                                               uint8_t const & level) const
{
  if (level < m_refineParams.level_max and indicator > m_refineParams.refine_th)
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_REFINE);
  else if (level > m_refineParams.level_min and indicator < m_refineParams.coarsen_th)
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_COARSEN);
  else
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_NOTHING);

} // ComputeRefineFlags<dim, device_t>::update_flag

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeRefineFlags<dim, device_t>::operator()(TagLohnerSplit const &,
                                              const index_t & global_index) const
{
  const auto iOct_global = global_index / m_userdata.num_cells();
  const auto cell_index = global_index - iOct_global * m_userdata.num_cells();

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_userdata.block_size());
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  const auto level = orchard_key_t<dim>::level(key_cur);
  auto       flag = AMRContextBase::KALYPSSO_FLAG_INIT;

  {
    auto indicator = compute_lohner_split<IX>(cell_loc, cell_index, m_refineParams.ivar);
    update_flag(flag, indicator, level);
  }

  {
    auto indicator = compute_lohner_split<IY>(cell_loc, cell_index, m_refineParams.ivar);
    update_flag(flag, indicator, level);
  }

  if constexpr (dim == 3)
  {
    auto indicator = compute_lohner_split<IZ>(cell_loc, cell_index, m_refineParams.ivar);
    update_flag(flag, indicator, level);
  }

  Kokkos::atomic_max(&m_flags(iOct_global), flag);

} // ComputeRefineFlags<dim, device_t>::operator()

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeRefineFlags<dim, device_t>::operator()(TagLohnerUnsplit const &,
                                              const index_t & global_index) const
{
  const auto iOct_global = global_index / m_userdata.num_cells();
  const auto cell_index = global_index - iOct_global * m_userdata.num_cells();

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_userdata.block_size());
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  const auto level = orchard_key_t<dim>::level(key_cur);
  auto       flag = AMRContextBase::KALYPSSO_FLAG_INIT;
  const auto indicator = compute_lohner_unsplit(cell_loc, cell_index, m_refineParams.ivar);

  update_flag(flag, indicator, level);

  Kokkos::atomic_max(&m_flags(iOct_global), flag);

} // ComputeRefineFlags<dim, device_t>::operator()

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeRefineFlags<dim, device_t>::operator()(TagSimpleGradient const &,
                                              const index_t & global_index) const
{
  const auto iOct_global = global_index / m_userdata.num_cells();
  const auto cell_index = global_index - iOct_global * m_userdata.num_cells();

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_userdata.block_size());
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  const auto level = orchard_key_t<dim>::level(key_cur);
  auto       flag = AMRContextBase::KALYPSSO_FLAG_INIT;
  const auto indicator = compute_simple_gradient(cell_loc, cell_index, m_refineParams.ivar);

  update_flag(flag, indicator, level);

  Kokkos::atomic_max(&m_flags(iOct_global), flag);

} // ComputeRefineFlags<dim, device_t>::operator()

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeRefineFlags<dim, device_t>::operator()(TagThresholdAffine const &,
                                              const index_t & global_index) const
{
  const auto iOct_global = global_index / m_userdata.num_cells();
  const auto cell_index = global_index - iOct_global * m_userdata.num_cells();

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_userdata.block_size());
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  const auto level = orchard_key_t<dim>::level(key_cur);
  auto       flag = AMRContextBase::KALYPSSO_FLAG_INIT;
  const auto indicator =
    compute_threshold_affine(cell_loc, cell_index, m_refineParams.ivar, m_refineParams.th_aff_par);

  if (level < m_refineParams.level_max and indicator > m_refineParams.th_aff_par.threshold)
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_REFINE);
  else if (level > m_refineParams.level_min and indicator < m_refineParams.th_aff_par.threshold)
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_COARSEN);
  else
    flag = KokkosExt::max_val(flag, AMRContextBase::KALYPSSO_DO_NOTHING);

  Kokkos::atomic_max(&m_flags(iOct_global), flag);

} // ComputeRefineFlags<dim, device_t>::operator()

// explicit template instantiation
template class ComputeRefineFlags<2, kalypsso::DefaultDevice>;
template class ComputeRefineFlags<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
