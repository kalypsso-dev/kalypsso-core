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
 * Enumerate types of equation of states (monofluid).
 */
BETTER_ENUM(EOS_TYPE, int,
            IDEAL_GAS,
            STIFFENED_GAS,
            VANDERWAALS_GAS,
            MIE_GRUNEISEN, // To be removed
            MIE_GRUNEISEN_SW, // shockwave eos
            MIE_GRUNEISEN_SW2, // shockwave eos (v2)
            MIE_GRUNEISEN_CC, // Cochran-Chan eos
            MIE_GRUNEISEN_JWL // JWL (Jones-Wilkins-Lee) eos
  )
// clang-format on


// clang-format off
/**
 * Enumerate types of equation of states (multifluid) using the Mie-Gruneisen form.
 */
BETTER_ENUM(MG_EOS_TYPE, int,
            MG_IDEAL_GAS =       0,
            MG_STIFFENED_GAS =   1,
            MG_VANDERWAALS_GAS = 2,
            MG_COCHRAN_CHAN =    3,
            MG_JWL =             4,
            MG_SHOCKWAVE =       5,
            MG_SHOCKWAVE2 =      6,
            MG_INVALID =         7, // MUST BE LAST
            MG_NUM = MG_INVALID     // total number of EOS
            )
// clang-format on

/**
 * Read eos type from input parameters file.
 *
 * \note This is only useful for monofluid simulation.
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
 * Read Mie-Gruneisen eos type from input parameters file.
 *
 * \note This is mostly useful for multifluid/multimaterial simulation.
 */
inline MG_EOS_TYPE
get_mg_eos_type(const size_t i_mat, ConfigMap const & config_map)
{
  const auto material_id = "material" + std::to_string(i_mat);

  auto eos_name = config_map.getString(material_id, "mg_eos_name", "MG_INVALID");
  auto maybe_value = MG_EOS_TYPE::_from_string_nothrow(eos_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return MG_EOS_TYPE::MG_INVALID;
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
