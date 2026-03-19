// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeVolumeIntegralValue.h
 *
 * \brief Contains the functor used to compute a volume integral value.
 */

#ifndef KALYPSSO_CORE_VOLUME_COMPUTE_INTEGRAL_VALUE_H_
#define KALYPSSO_CORE_VOLUME_COMPUTE_INTEGRAL_VALUE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

namespace core
{

/**
 * \class ComputeVolumeIntegralValue
 * \brief Functor wrapper that computes a volume integral value over a data array.
 *
 * \tparam dim The dimension of the problem (must be 2 or 3).
 * \tparam device_t On which Kokkos device to run the underlying functor.
 *
 */
template <size_t dim, typename device_t>
class ComputeVolumeIntegralValue
{
public:
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  /**
   * \brief Computes a volume integral value over a data array.
   *
   * Volume integral value is simply defined as the sum of all values of an array (can be a
   * DataArrayBlock), but it could be any other integral value if needed.
   *
   * \param data Values array of the simulation.
   * \param start_octant The first octant to compute.
   * \param end_octant The last octant to compute, excluded.
   * \param keys Orchard keys.
   * \param var_index The variable to sum on.
   * \param config_map Inputted config map.
   * \param par_env Parallel environment.
   *
   * \note par_env is only used when the number of MPI processes is strictly larger than 1.
   *
   * \returns The volume integral value.
   */
  static real_t
  apply(const DataArrayBlock<dim, real_t, device_t> & data,
        const int32_t                                 start_octant,
        const int32_t                                 end_octant,
        const OrchardKeys &                           keys,
        const int32_t                                 var_index,
        const ConfigMap &                             config_map,
        [[maybe_unused]] const ParallelEnv &          par_env);

  /**
   * \brief Kokkos kernel.
   *
   * \param i_global The global index of the cell.
   * \param total The summed total.
   */
  KOKKOS_FUNCTION void
  operator()(const int32_t i_global, real_t & total) const;

private:
  //! Kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  /**
   * \brief Constructor.
   *
   * \param data Values array of the simulation.
   * \param keys Orchard keys.
   * \param var_index The variable to sum on.
   * \param config_map Inputted config map.
   */
  ComputeVolumeIntegralValue(const DataArrayBlock<dim, real_t, device_t> & data,
                             const OrchardKeys &                           keys,
                             const int32_t                                 var_index,
                             const ConfigMap &                             config_map);

  //! Some input data
  DataArrayBlock<dim, real_t, device_t> m_data;

  //! The Orchard keys
  OrchardKeys m_keys;

  //! The variable index to use for integral value
  int32_t m_var_index;

  //! Tree scaling factor (used for computing local metric)
  real_t m_scaling_factor;
};

extern template class ComputeVolumeIntegralValue<2, kalypsso::DefaultDevice>;
extern template class ComputeVolumeIntegralValue<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_VOLUME_COMPUTE_INTEGRAL_VALUE_H_
