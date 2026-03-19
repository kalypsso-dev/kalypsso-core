// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConformalNeighborStatus.h
 *
 */
#ifndef KALYPSSO_CORE_CONFORMALNEIGHBORSTATUS_H_
#define KALYPSSO_CORE_CONFORMALNEIGHBORSTATUS_H_

#include <Kokkos_Macros.hpp>          // for KOKKOS_ENABLE_XXX
#include <kalypsso/core/mesh_utils.h> // for Face::XMIN, etc...

namespace kalypsso
{

struct conformal_neighbor_status
{

  enum neighbor_status : uint8_t
  {
    NEIGHBOR_IS_AT_SAME_LEVEL = 0,
    NEIGHBOR_IS_FINER = 1,
    NEIGHBOR_IS_COARSER = 2,
    NEIGHBOR_IS_UNAVAILABLE = 3
  };

  KOKKOS_INLINE_FUNCTION static bool
  is_at_same_level(uint8_t const & status)
  {
    return status == NEIGHBOR_IS_AT_SAME_LEVEL;
  }

  KOKKOS_INLINE_FUNCTION static bool
  is_finer(uint8_t const & status)
  {
    return status == NEIGHBOR_IS_FINER;
  }

  KOKKOS_INLINE_FUNCTION static bool
  is_coarser(uint8_t const & status)
  {
    return status == NEIGHBOR_IS_COARSER;
  }

  KOKKOS_INLINE_FUNCTION static bool
  is_conformal(uint8_t const & status)
  {
    return is_at_same_level(status);
  }

  KOKKOS_INLINE_FUNCTION static bool
  is_non_conformal(uint8_t const & status)
  {
    return is_finer(status) or is_coarser(status);
  }

  KOKKOS_INLINE_FUNCTION static bool
  is_non_available(uint8_t const & status)
  {
    return status == NEIGHBOR_IS_UNAVAILABLE;
  }
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_CONFORMALNEIGHBORSTATUS_H_
