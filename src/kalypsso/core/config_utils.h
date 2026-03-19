// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * config_utils.h
 */
#ifndef KALYPSSO_CORE_CONFIG_UTILS_H_
#define KALYPSSO_CORE_CONFIG_UTILS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/utils_block.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/p4est/connectivity.h> // for enum connectivity_periodic_t

namespace kalypsso
{

// ===========================================================
// ===========================================================
/**
 * read block sizes from config map.
 */
template <size_t dim>
auto
get_block_sizes(ConfigMap const & config_map) -> block_size_t<dim>
{

  block_size_t<dim> block_sizes;

  block_sizes[IX] = config_map.getInteger("amr", "bx", 0);
  block_sizes[IY] = config_map.getInteger("amr", "by", 0);

  if constexpr (dim == 3)
  {
    block_sizes[IZ] = config_map.getInteger("amr", "bz", 1);
  }

  return block_sizes;

} // get_block_sizes

// ===========================================================
// ===========================================================
/**
 * Increase a block size for holding flux.
 */
template <size_t dim>
auto
get_flux_block_sizes(block_size_t<dim> block_sizes, int direction = IX) -> block_size_t<dim>
{

  // flux block size
  auto flux_block_sizes = block_sizes;

  // direction must be smaller than to be valid
  if (static_cast<uint32_t>(direction) < dim)
  {
    flux_block_sizes[direction]++;
  }

  return flux_block_sizes;

} // get_flux_block_sizes

} // namespace kalypsso

#endif // KALYPSSO_CORE_CONFIG_UTILS_H_
