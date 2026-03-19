// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConservativityCheck.cpp
 *
 * \brief Contains a helper class for checking conservativity.
 */

#include <kalypsso/core/ConservativityCheck.h>
#include <kalypsso/core/ComputeVolumeIntegralValue.h>

#include <iostream>
#include <iomanip>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ConservativityCheck<dim, device_t>::register_value(
  const DataArrayBlock<dim, real_t, device_t> & data,
  const int32_t                                 start_octant,
  const int32_t                                 end_octant,
  const OrchardKeys &                           keys,
  const int32_t                                 var_index,
  const std::string                             var_name,
  const ConfigMap &                             config_map,
  const ParallelEnv &                           par_env,
  bool                                          is_reference)
{

  const auto value = ComputeVolumeIntegralValue<dim, device_t>::apply(
    data, start_octant, end_octant, keys, var_index, config_map, par_env);

  insert_or_update_value(var_name, value, is_reference);

} // ConservativityCheck<dim, device_t>::register_value

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ConservativityCheck<dim, device_t>::print_report(const ParallelEnv & par_env) const
{
  if (m_reference_values.size() > 0)
  {
    if (par_env.rank() == 0)
    {
      std::cout << "========================================\n";
      std::cout << "Conservativity check report             \n";
      std::cout << "========================================\n";
    }
  }

  for (auto ref : m_reference_values)
  {
    auto const & var_name = ref.first;
    auto const & ref_value = ref.second;

    auto it = m_test_values.find(var_name);

    if (it != m_test_values.end())
    {
      // we found a value that exists in both reference and test map
      auto const & test_value = it->second;

      const auto diff = fabs(test_value - ref_value);
      const auto refv = fabs(ref_value);

      if (par_env.rank() == 0)
      {

#ifdef KALYPSSO_CORE_USE_SPDLOG
        fmt::print("Conservativity check of {}\n", var_name);
        fmt::print("{: >30} {}\n", "Initial volume integral: ", ref_value);
        fmt::print("{: >30} {}\n", "Final   volume integral: ", test_value);
        fmt::print("{: >30} {}\n", "L1 absolute error: ", diff);
        if (refv > SMALL_VALUE)
        {
          fmt::print("{: >30} {}\n", "L1 relative error: ", diff / refv);
        }
        fmt::print("-------------------------------------\n");
#else
        // clang-format off
        std::cout << "Conservativity check of " << var_name << "\n";
        std::cout << std::setprecision(10)
                  << std::setw(30) << "Initial volume integral: " << ref_value << "\n"
                  << std::setw(30) << "Final   volume integral: " << test_value << "\n"
                  << std::setw(30) << "L1 absolute error: " << diff << "\n";
        if (refv > SMALL_VALUE)
        {
          std::cout << std::setw(30) << "L1 relative error: " << diff / refv  << "\n";
        }
        std::cout << "-------------------------------------\n";
        // clang-format on
#endif
      } // end rank == 0
    }
    else
    {
      // we should print some error message, since a value exist in reference map but no
      // corresponding value in test map
      if (par_env.rank() == 0)
      {
        std::cerr << "variable " << var_name
                  << " was not found in test values map; check you code for a missing call to "
                     "register_value\n";
      }
    }
  } // end for variables

  if (m_reference_values.size() > 0)
  {
    if (par_env.rank() == 0)
    {
      std::cout << "========================================\n";
    }
  }

} // ConservativityCheck<dim, device_t>::print_report

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
ConservativityCheck<dim, device_t>::insert_or_update_value(std::string var_name,
                                                           real_t      value,
                                                           bool        is_reference)
{
  auto & value_map = is_reference ? m_reference_values : m_test_values;

  auto it = value_map.find(var_name);

  if (it != value_map.end())
  {
    // entry already exists, just update value
    value_map.at(var_name) = value;
  }
  else
  {
    value_map.insert({ var_name, value });
  }
} // ConservativityCheck<dim, device_t>::insert_or_update_value

// ================================================================================================
// ================================================================================================
template class ConservativityCheck<2, kalypsso::DefaultDevice>;
template class ConservativityCheck<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
