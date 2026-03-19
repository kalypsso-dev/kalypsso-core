// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeSchlieren.cpp
 */
#include <kalypsso/core/ComputeSchlieren.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
ComputeSchlieren<dim, device_t>::run(ConfigMap const &                    config_map,
                                     [[maybe_unused]] const ParallelEnv & par_env,
                                     amr_hashmap_t                        amr_hashmap,
                                     orchard_key_view_t                   orchard_keys,
                                     int32_t                              local_num_octants,
                                     block_size_t<dim>                    block_sizes,
                                     brick_size_t<dim>                    brick_sizes,
                                     Kokkos::Array<bool, dim>             is_brick_periodic,
                                     DataArrayBlock_t userdata) -> DataArrayBlock_t
{
  KOKKOS_ASSERT(userdata.shape()[IX] > 2 && "userdata has a wrong shape, block size too small");
  KOKKOS_ASSERT(userdata.shape()[IY] > 2 && "userdata has a wrong shape, block size too small");
  if constexpr (dim == 3)
  {
    KOKKOS_ASSERT(userdata.shape()[IZ] > 2 && "userdata has a wrong shape, block size too small");
  }

  auto schlieren = DataArrayBlock_t("Schlieren", userdata.block_size(), 1, local_num_octants);

  ComputeSchlieren<dim, device_t> functor(config_map,
                                          amr_hashmap,
                                          orchard_keys,
                                          local_num_octants,
                                          block_sizes,
                                          brick_sizes,
                                          is_brick_periodic,
                                          get_scaling_factor(config_map),
                                          userdata,
                                          schlieren);

  const auto nbCellsPerLeaf = Kokkos::dim_prod(block_sizes);
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  real_t              local_max_norm_grad = 0;
  Kokkos::Max<real_t> reducer(local_max_norm_grad);

  //
  // 1. compute max of norm of gradient and reduce to have the global maximum
  //
  Kokkos::parallel_reduce("compute_max_norm_of_gradient",
                          Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                          functor,
                          reducer);

  real_t max_norm_grad = 0;
#ifdef KALYPSSO_CORE_USE_MPI
  par_env.comm().MPI_Allreduce<MpiComm::MAX>(&local_max_norm_grad, &max_norm_grad, 1);
#else
  max_norm_grad = local_max_norm_grad;
#endif

  // divide by max norm of gradient, and apply scaling if required
  const auto schlieren_scaling_type = read_schlieren_scaling(config_map);

  //
  // 2. actually compute schlieren
  //
  if (schlieren_scaling_type == +SCHLIEREN_SCALING::SQRT)
  {
    Kokkos::parallel_for(
      "compute_schlieren_sqrt",
      Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
      KOKKOS_LAMBDA(int32_t global_index) {
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        schlieren(cell_index, 0, iOct) = sqrt(schlieren(cell_index, 0, iOct) / max_norm_grad);
      });
  }
  else if (schlieren_scaling_type == +SCHLIEREN_SCALING::LOG)
  {
    Kokkos::parallel_for(
      "compute_schlieren_log",
      Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
      KOKKOS_LAMBDA(int32_t global_index) {
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        schlieren(cell_index, 0, iOct) = schlieren(cell_index, 0, iOct) > 0
                                           ? log(schlieren(cell_index, 0, iOct) / max_norm_grad)
                                           : ZERO_F;
      });
  }
  else if (schlieren_scaling_type == +SCHLIEREN_SCALING::EXP)
  {
    auto schlieren_params = SchlierenParams(config_map);

    Kokkos::parallel_for(
      "compute_schlieren_exp",
      Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
      KOKKOS_LAMBDA(int32_t global_index) {
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        const auto k =
          userdata(cell_index, schlieren_params.ivar_vol_frac, iOct) > schlieren_params.vol_frac_th
            ? schlieren_params.scale_high
            : schlieren_params.scale_low;
        schlieren(cell_index, 0, iOct) = exp(-k * schlieren(cell_index, 0, iOct) / max_norm_grad);
      });
  }
  else
  {
    Kokkos::parallel_for(
      "compute_schlieren_default",
      Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
      KOKKOS_LAMBDA(int32_t global_index) {
        const auto iOct = global_index / nbCellsPerLeaf;
        const auto cell_index = global_index - iOct * nbCellsPerLeaf;

        schlieren(cell_index, 0, iOct) /= max_norm_grad;
      });
  }

  return schlieren;

}; // ComputeSchlieren<dim, device_t>::run

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeSchlieren<dim, device_t>::getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                                                          int32_t                   cell_index,
                                                          shift_t<dim>              shift,
                                                          int                       ivar,
                                                          bool & is_neighbor_not_owned) const
{

  const auto cell_loc_neigh = m_helper.getNeighLoc(cell_loc, shift);
  const auto cell_index_neigh = coord_to_cellindex<dim>(cell_loc_neigh.ijk, m_block_sizes);

  if (cell_loc_neigh.is_outside_domain or cell_loc_neigh.iOct >= m_local_num_octants)
  {
    // neighbor is not a locally owned quadrant
    is_neighbor_not_owned = true;

    // return value in current cell
    return m_userdata_in(cell_index, ivar, cell_loc.iOct);
  }
  else
  {
    // if neighbor is at same level or is coarser, just use the value
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

} // ComputeSchlieren<dim, device_t>::getNeighborDataSameLevel

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <int32_t dir>
KOKKOS_FUNCTION real_t
ComputeSchlieren<dim, device_t>::compute_gradient_along_dir(CellLocation_t const & cell_loc,
                                                            int32_t const &        cell_index,
                                                            int const &            ivar) const
{
  const auto & b = m_userdata_in.block_size();

  bool is_neighbor_not_owned = false;

  const auto data_L = getNeighborDataSameLevel(
    cell_loc, cell_index, get_shift_left<dim, dir>(), ivar, is_neighbor_not_owned);

  const auto data_R = getNeighborDataSameLevel(
    cell_loc, cell_index, get_shift_right<dim, dir>(), ivar, is_neighbor_not_owned);

  const auto dx = compute_cell_length<dim>(cell_loc.key, b[dir]) * m_scaling_factor;

  return is_neighbor_not_owned ? (data_R - data_L) / dx : (data_R - data_L) / (2 * dx);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION real_t
ComputeSchlieren<dim, device_t>::compute_norm_of_gradient(CellLocation_t const & cell_loc,
                                                          int32_t const &        cell_index,
                                                          int const &            ivar) const
{
  const auto gradx = compute_gradient_along_dir<IX>(cell_loc, cell_index, ivar);
  const auto grady = compute_gradient_along_dir<IY>(cell_loc, cell_index, ivar);

  auto grad_norm = gradx * gradx + grady * grady;

  if constexpr (dim == 3)
  {
    const auto gradz = compute_gradient_along_dir<IZ>(cell_loc, cell_index, ivar);
    grad_norm += gradz * gradz;
  }

  return sqrt(grad_norm);

} // ComputeSchlieren<dim, device_t>::compute_norm_of_gradient

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeSchlieren<dim, device_t>::operator()(const index_t & global_index,
                                            real_t &        max_norm_grad) const
{
  const auto iOct_global = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct_global * m_nbCellsPerLeaf;

  const auto     coord = cellindex_to_coord<dim>(cell_index, m_block_sizes);
  const auto     key_cur = m_orchard_keys_device(iOct_global);
  CellLocation_t cell_loc{ coord, key_cur, iOct_global, false };

  auto norm_grad = compute_norm_of_gradient(cell_loc, cell_index, m_schlieren_params.ivar_rho);

  max_norm_grad = fmax(max_norm_grad, norm_grad);

  m_userdata_out(cell_index, 0, iOct_global) = norm_grad;

} // ComputeSchlieren<dim, device_t>::operator()

// explicit template instantiation
template class ComputeSchlieren<2, kalypsso::DefaultDevice>;
template class ComputeSchlieren<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
