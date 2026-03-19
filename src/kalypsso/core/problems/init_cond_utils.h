// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file init_cond_utils.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_INITCONDUTILS_H_
#define KALYPSSO_CORE_PROBLEMS_INITCONDUTILS_H_

#include <cstdint>
#include <../../better-enums/enum.h>

// forward declaration
class ConfigMap;

namespace kalypsso
{

namespace core
{
// clang-format off
/**
 * An enum type to represent all possible refine indicator types used during initial condition.
 *
 * A given test case may only support a subset of this enum.
 * The default value is to use the same refine criterion as the regular dynamics (t>0).
 */
BETTER_ENUM(InitConditionsIndicator, uint8_t,
            ALWAYS_REFINE = 0,             // always refine up to level max
            GEOMETRIC = 1,                 // refine near interface
            SAME_AS_REGULAR_DYNAMICS = 2   // use the same refine criterion as specified with
                                           // amr/refine_criterion in ini inputfile
           )
// clang-format on

/**
 * Retrieve refine criterion type to use inside initial conditions.
 */
InitConditionsIndicator
get_init_indicator(ConfigMap const & config_map);

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_INITCONDUTILS_H_
