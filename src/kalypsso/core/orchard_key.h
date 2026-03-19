// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file orchard_key.h
 * \brief
 * Design a 64-bit "key" type that can be used as an entry to our adaptive mesh refinement
 * hash-table data structure.
 *
 * This is adapted from reference:
 * An octree-based, cartesian navier–stokes solver for modern cluster architectures,
 * Dylan Jude, Jay Sitaraman and, Andrew Wissink, The journal of Supercomputing, 78 (3),
 * June 2022.
 *
 * This article only describe 3D; here we also define an equivalent for 2D. See below orchard_key_t.
 *
 */
#ifndef KALYPSSO_CORE_ORCHARD_KEY_H
#define KALYPSSO_CORE_ORCHARD_KEY_H

#include <kalypsso/core/enums.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/BitFieldInteger.h>
#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/kalypsso_macros.h>

#include <cstdint>

#include <kalypsso/core/orchard_key_base.h>

namespace kalypsso
{

/**
 * \struct orchard_key_t
 *
 * Define type used as a key to a Kokkos::UnorderedMap data structure, adapted from
 * Jude etal, J. of Supercomputing, 78(3), June 2022.
 *
 * Bit signification:
 *
 * ========================================================================
 * 2D:
 *
 *   12 bits       44 bits         6 bits           2 bits
 * <---------><-----------------><----------><------------------>
 *   treeId       Morton Id         level       outside status
 *
 * This means
 * - up to 2^6=64 trees in each direction (X,Y)
 * - up to 23 (from 0 to 22) levels of refinement
 * - level is encode on 6 bits (more than enough)
 * - outside status is encoded on 2 bits, 1 bit for each type of face; e.g.
 *   a bulk quadrant will be encode as 00, a quadrant touching only face will be either 01 or 10,
 * and a corner will be encoded as 11.
 *
 * ========================================================================
 * 3D:
 *
 *   15 bits       42 bits         4 bits           3 bits
 * <---------><-----------------><----------><------------------>
 *   treeId       Morton Id         level       outside status
 *
 * This means
 * - up to 2^5=32 trees in each direction (X,Y,Z)
 * - up to 15 (from 0 14) levels of refinement
 * - level is encoded on 4 bits (just enough)
 * - outside status is encoded on 3 bits, 1 bit for each type of face; e.g.
 *   a bulk quadrant will be encode as 000, a quadrant touching only face will be either 001 or 010
 * or 100, en edge by 011, 101, or 110 and a corner will be encoded as 111.
 */
template <size_t dim>
struct orchard_key_t;

#include "orchard_key_impl_2d.h"
#include "orchard_key_impl_3d.h"

} // namespace kalypsso

#endif // KALYPSSO_CORE_ORCHARD_KEY_H
