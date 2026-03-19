// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * physical_constants.h
 */
#ifndef KALYPSSO_CORE_PHYSICAL_CONSTANTS_H_
#define KALYPSSO_CORE_PHYSICAL_CONSTANTS_H_

#include <kalypsso/core/real_type.h>

namespace kalypsso
{

namespace constants
{

/**
 * The Avogadro and Boltzmann constants are slightly different from the exact values
 * defined by NIST since 2019:
 * - https://physics.nist.gov/cgi-bin/cuu/Value?na|search_for=avogadro
 * - https://physics.nist.gov/cgi-bin/cuu/Value?k
 *
 * \todo see if it would be interesting support also the exact values.
 */

static constexpr real_t AVOGADRO = KALYPSSO_NUM(6.02214179e23);  //!< Avogadro constant
static constexpr real_t BOLTZMANN = KALYPSSO_NUM(1.3806488e-23); //!< Boltzmann constant
static constexpr real_t MU0 = 4 * PI_F * KALYPSSO_NUM(1e-7);     //!< vacuum magnetic permeability

} // namespace constants

} // namespace kalypsso

#endif // KALYPSSO_CORE_PHYSICAL_CONSTANTS_H_
