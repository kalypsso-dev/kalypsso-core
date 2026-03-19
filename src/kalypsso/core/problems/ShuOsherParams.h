// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ShuOsherParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_SHUOSHER_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_SHUOSHER_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Shu-Osher test parameters.
 *
 * A moving shock interacting with a density sine wave.
 *
 * Reference:
 * Efficient implementation of essentially non-oscillatory shock-capturing schemes, II,
 * C.-W. Shu and S. Osher, JCP vol. 83, pp 32-78 (1989).
 * https://doi.org/10.1016/0021-9991(89)90222-2
 */
struct ShuOsherParams
{

  // Shu-Osher problem parameters: left and right state
  real_t rhoL;
  real_t uL;
  real_t pL;

  real_t rhoR;
  real_t uR;
  real_t pR;

  real_t A;  // density wave amplitude
  real_t k;  // density wave number
  real_t x0; // initial position of interface

  ShuOsherParams(ConfigMap const & config_map)
    : rhoL(config_map.getReal("shu_osher", "rhoL", KALYPSSO_NUM(27.0) / 7))
    , uL(config_map.getReal("shu_osher", "uL", 4 * sqrt(KALYPSSO_NUM(35.0)) / KALYPSSO_NUM(9.0)))
    , pL(config_map.getReal("shu_osher", "pL", KALYPSSO_NUM(31.0) / 3))
    , rhoR(config_map.getReal("shu_osher", "rhoR", KALYPSSO_NUM(1.0)))
    , uR(config_map.getReal("shu_osher", "uR", KALYPSSO_NUM(0.0)))
    , pR(config_map.getReal("shu_osher", "pR", KALYPSSO_NUM(1.0)))
    , A(config_map.getReal("shu_osher", "A", KALYPSSO_NUM(0.2)))
    , k(config_map.getReal("shu_osher", "k", KALYPSSO_NUM(5.0)))
    , x0(config_map.getReal("shu_osher", "x0", KALYPSSO_NUM(-4.0)))
  {}

}; // struct ShuOsherParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_SHUOSHER_PARAMS_H_
