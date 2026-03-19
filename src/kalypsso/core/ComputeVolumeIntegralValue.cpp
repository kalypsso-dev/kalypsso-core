// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeVolumeIntegralValue.cpp
 *
 * \brief Contains the definition of ComputeVolumeIntegralValue.
 */

#include <kalypsso/core/ComputeVolumeIntegralValue.h>

#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
ComputeVolumeIntegralValue<dim, device_t>::ComputeVolumeIntegralValue(
  const DataArrayBlock<dim, real_t, device_t> & data,
  const OrchardKeys &                           keys,
  const int32_t                                 var_index,
  const ConfigMap &                             config_map)
  : m_data(data)
  , m_keys(keys)
  , m_var_index(var_index)
  , m_scaling_factor(get_scaling_factor(config_map))
{}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
real_t
ComputeVolumeIntegralValue<dim, device_t>::apply(const DataArrayBlock<dim, real_t, device_t> & data,
                                                 const int32_t                        start_octant,
                                                 const int32_t                        end_octant,
                                                 const OrchardKeys &                  keys,
                                                 const int32_t                        var_index,
                                                 const ConfigMap &                    config_map,
                                                 [[maybe_unused]] const ParallelEnv & par_env)
{
  ComputeVolumeIntegralValue<dim, device_t> functor(data, keys, var_index, config_map);

  const int32_t                       nb_cells = data.num_cells();
  const int32_t                       start = start_octant * nb_cells;
  const int32_t                       end = end_octant * nb_cells;
  Kokkos::RangePolicy<ExecutionSpace> policy(start, end);

  real_t              local_total = 0;
  Kokkos::Sum<real_t> reducer(local_total);
  Kokkos::parallel_reduce("kalypsso::core::ComputeVolumeIntegralValue", policy, functor, reducer);

  real_t global_total = local_total;
#ifdef KALYPSSO_CORE_USE_MPI
  par_env.comm().template MPI_Allreduce<MpiComm::SUM>(&local_total, &global_total, 1);
#endif // KALYPSSO_CORE_USE_MPI

  return global_total;
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
ComputeVolumeIntegralValue<dim, device_t>::operator()(const int32_t i_global, real_t & total) const
{
  const auto block_size = m_data.block_size();
  const auto nb_cells = m_data.num_cells();
  const auto i_oct = i_global / nb_cells;
  const auto i_cell = i_global - nb_cells * i_oct;

  const auto level = orchard_key_t<dim>::level(m_keys(i_oct));
  const auto dx = compute_cell_length<dim>(level, block_size[IX]) * m_scaling_factor;

  real_t vol = dx * dx;
  if constexpr (dim == 3)
    vol *= dx;

  total += vol * m_data(i_cell, m_var_index, i_oct);
}

// ================================================================================================
// ================================================================================================
template class ComputeVolumeIntegralValue<2, kalypsso::DefaultDevice>;
template class ComputeVolumeIntegralValue<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
