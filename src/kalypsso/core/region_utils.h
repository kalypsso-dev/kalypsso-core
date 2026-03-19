// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * \file region_utils.h
 */
#ifndef KALYPSSO_CORE_REGION_UTILS_H_
#define KALYPSSO_CORE_REGION_UTILS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>

#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/utils_block.h>

namespace kalypsso
{

// ==========================================================================
// ==========================================================================
/**
 * Return true if cell is pure.
 *
 * A cell is considered pure if all corners belong to the same region.
 *
 * \param[in] regions array of regions (one per corner)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_cell_fully_inside_region(Kokkos::Array<int, Corner::num_corners<dim>()> const & regions)
{
  const auto region = regions[0];
  for (uint8_t i_corner = 1; i_corner < Corner::num_corners<dim>(); i_corner++)
  {
    if (region != regions[i_corner])
      return false;
  }
  return true;
}

// ==========================================================================
// ==========================================================================
/**
 * Return true if cell partially overlaps with a given region.
 *
 * \param[in] regions array of regions (one per corner)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
does_cell_overlap_with_region(Kokkos::Array<int, Corner::num_corners<dim>()> const & regions,
                              int                                                    region)
{
  for (uint8_t i_corner = 0; i_corner < Corner::num_corners<dim>(); i_corner++)
  {
    if (region == regions[i_corner])
      return true;
  }
  return false;
}

} // namespace kalypsso

#endif // KALYPSSO_CORE_REGION_UTILS_H_
