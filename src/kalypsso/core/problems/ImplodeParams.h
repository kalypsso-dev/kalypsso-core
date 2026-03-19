// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ImplodeParams.h
 *
 * Hydrodynamical Implosion Test.
 *
 * Implosion test references:
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 * - https://www.sciencedirect.com/science/article/pii/S0021999199962952
 * - http://www-troja.fjfi.cvut.cz/~liska/CompareEuler/compare8/
 * - https://www.sciencedirect.com/science/article/pii/S0045793021003364#b46
 *
 * \note It can be used for both hydro and MHD test cases.
 */
#ifndef KALYPSSO_CORE_PROBLEMS_IMPLODE_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_IMPLODE_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

#include <../../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
BETTER_ENUM(ImplodeShape, int, DIAGONAL, DIAMOND)
// clang-format on

/**
 * Implode test parameters.
 */
struct ImplodeParams
{

  //! get shape
  static inline ImplodeShape
  get_shape(ConfigMap const & config_map)
  {
    auto shapeStr = config_map.getString("implode", "shape", "DIAGONAL");
    std::transform(shapeStr.begin(), shapeStr.end(), shapeStr.begin(), ::toupper);

    auto maybe_value = ImplodeShape::_from_string_nothrow(shapeStr.c_str());
    if (maybe_value)
    {
      return *maybe_value;
    }

    // default
    return ImplodeShape::DIAGONAL;
  }

  // outer parameters
  real_t rho_out;
  real_t p_out;
  real_t u_out;
  real_t v_out;
  real_t w_out;
  real_t Bx_out;
  real_t By_out;
  real_t Bz_out;

  // inner parameters
  real_t rho_in;
  real_t p_in;
  real_t u_in;
  real_t v_in;
  real_t w_in;
  real_t Bx_in;
  real_t By_in;
  real_t Bz_in;

  //! shape: diagonal or diamond
  ImplodeShape shape;

  // if true slightly change init condition to be non trivial
  // gradient along domain diagonal
  bool debug;

  ImplodeParams(ConfigMap const & config_map)
    : rho_out(config_map.getReal("implode", "rho_out", KALYPSSO_NUM(1.0)))
    , p_out(config_map.getReal("implode", "p_out", KALYPSSO_NUM(1.0)))
    , u_out(config_map.getReal("implode", "u_out", KALYPSSO_NUM(0.0)))
    , v_out(config_map.getReal("implode", "v_out", KALYPSSO_NUM(0.0)))
    , w_out(config_map.getReal("implode", "w_out", KALYPSSO_NUM(0.0)))
    , Bx_out(config_map.getReal("implode", "Bx_out", KALYPSSO_NUM(0.0)))
    , By_out(config_map.getReal("implode", "By_out", KALYPSSO_NUM(0.0)))
    , Bz_out(config_map.getReal("implode", "Bz_out", KALYPSSO_NUM(0.0)))
    , rho_in(config_map.getReal("implode", "rho_in", KALYPSSO_NUM(0.125)))
    , p_in(config_map.getReal("implode", "p_in", KALYPSSO_NUM(0.14)))
    , u_in(config_map.getReal("implode", "u_in", KALYPSSO_NUM(0.0)))
    , v_in(config_map.getReal("implode", "v_in", KALYPSSO_NUM(0.0)))
    , w_in(config_map.getReal("implode", "w_in", KALYPSSO_NUM(0.0)))
    , Bx_in(config_map.getReal("implode", "Bx_in", KALYPSSO_NUM(0.0)))
    , By_in(config_map.getReal("implode", "By_in", KALYPSSO_NUM(0.0)))
    , Bz_in(config_map.getReal("implode", "Bz_in", KALYPSSO_NUM(0.0)))
    , shape(get_shape(config_map))
    , debug(config_map.getBool("implode", "debug", false))
  {}

}; // struct ImplodeParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_IMPLODE_PARAMS_H_
