// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StiffenedGasEos.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_UTILS_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_UTILS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/config/ConfigMap.h>

#include <../../better-enums/enum.h>

namespace kalypsso
{

namespace core
{

namespace eos
{

//! only useful for ideal gas EOS
KOKKOS_INLINE_FUNCTION real_t
gamma6(real_t const & gamma0)
{
  return (gamma0 + ONE_F) / (TWO_F * gamma0);
}

// clang-format off
/**
 * Enumerate types of stencil used for finite difference estimation first derivative.
 */
BETTER_ENUM(EOS_TYPE, int,
            IDEAL_GAS,
            STIFFENED_GAS,
            VANDERWAALS_GAS,
            MIE_GRUNEISEN)
// clang-format on

/**
 * Read eos type from input parameters file.
 */
inline EOS_TYPE
get_eos_type(ConfigMap const & config_map)
{
  auto eos_name = config_map.getString("eos", "name", "STIFFENED_GAS");
  auto maybe_value = EOS_TYPE::_from_string_nothrow(eos_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return EOS_TYPE::STIFFENED_GAS;
}

/**
 * Read specific heat ratio.
 */
inline auto
get_gamma(const size_t i_mat, const ConfigMap & config_map)
{
  const auto material_id = "material" + std::to_string(i_mat);

  return config_map.getReal(material_id, "gamma", KALYPSSO_NUM(1.4));
}

/**
 * Read pinf (stiffened gas parameter).
 */
inline auto
get_pinf(const size_t i_mat, const ConfigMap & config_map)
{
  const auto material_id = "material" + std::to_string(i_mat);

  return config_map.getReal(material_id, "pinf", KALYPSSO_NUM(0.0));
}

} // namespace eos

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_UTILS_H_
