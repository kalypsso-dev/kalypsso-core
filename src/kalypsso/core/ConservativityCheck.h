// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConservativityCheck.h
 *
 * \brief Contains a helper class for checking conservativity.
 *
 * This class is only really useful when solving a system of conservation laws (like Euler) using
 * either periodic or wall border conditions for which we know that all volume integral value of
 * conservative variables should remain constant in time. This class register the volume integral
 * values at initial and final time and provide a method to display relative error.
 */
#ifndef KALYPSSO_CORE_CONSERVATIVITY_CHECK_H_
#define KALYPSSO_CORE_CONSERVATIVITY_CHECK_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <map>
#include <string>

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
class ConservativityCheck
{
public:
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  static constexpr real_t SMALL_VALUE = KALYPSSO_NUM(1e-13);

  /**
   * \brief ConservativityCheck constructor.
   *
   */
  ConservativityCheck()
    : m_reference_values()
    , m_test_values(){};

  /**
   * \brief Register a reference or test value.
   *
   * \param data Values array of the simulation.
   * \param start_octant The first octant to compute.
   * \param end_octant The last octant to compute, excluded.
   * \param keys Orchard keys.
   * \param var_index The variable to sum on.
   * \param config_map Inputted config map.
   * \param is_reference Set to true if you want to register a reference value else a test value.
   */
  void
  register_value(const DataArrayBlock<dim, real_t, device_t> & data,
                 const int32_t                                 start_octant,
                 const int32_t                                 end_octant,
                 const OrchardKeys &                           keys,
                 const int32_t                                 var_index,
                 const std::string                             var_name,
                 const ConfigMap &                             config_map,
                 const ParallelEnv &                           par_env,
                 bool                                          is_reference);

  /**
   * \brief Print conservativity check report.
   *
   * Compare reference and test values; then display relative errors.
   */
  void
  print_report(const ParallelEnv & par_env) const;

private:
  //! volume integral reference values (usually computed at initial time)
  std::map<std::string, real_t> m_reference_values;

  //! volume integral test values to be compared with reference values (usually computed at final
  //! time)
  std::map<std::string, real_t> m_test_values;

  //! helper method to insert or update (var_name,value) in either reference or test map
  void
  insert_or_update_value(std::string var_name, real_t value, bool is_reference);

}; // class ConservativityCheck

extern template class ConservativityCheck<2, kalypsso::DefaultDevice>;
extern template class ConservativityCheck<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_CONSERVATIVITY_CHECK_H_
