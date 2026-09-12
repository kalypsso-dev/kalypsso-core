// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FourQuadrantParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * Four-Quadrant problem test parameters.
 */
struct FourQuadrantParams
{

  Kokkos::Array<real_t, 3> pos;           //! Discontinutity locations
  int                      config_number; //! Config number

  FourQuadrantParams(ConfigMap const & config_map)
  {
    pos[0] = config_map.getReal("four_quadrant", "x", KALYPSSO_NUM(0.8));
    pos[1] = config_map.getReal("four_quadrant", "y", KALYPSSO_NUM(0.8));
    pos[2] = config_map.getReal("four_quadrant", "z", KALYPSSO_NUM(0.8));
    config_number = config_map.getInteger("four_quadrant", "config_number", 0);
  }

}; // struct FourQuadrantParameters

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_FOURQUADRANT_PARAMS_H_
