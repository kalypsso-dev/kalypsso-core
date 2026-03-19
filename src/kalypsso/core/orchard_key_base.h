// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file orchard_key_base.h
 */
#ifndef KALYPSSO_CORE_ORCHARD_KEY_BASE_H
#define KALYPSSO_CORE_ORCHARD_KEY_BASE_H

#include <cstdint>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/brick_base.h>

namespace kalypsso
{

//! base type used internally in orchard key implementation
using key_t = uint64_t;

//! type alias for storing octant memory index in hashmap
using iOct_t = int64_t;


/**
 * Kind of namespace struct used to define type alias for Kokkos view of orchard keys.
 */
template <typename device_t>
struct orchard_key_base_t
{
  using view_t = typename Kokkos::View<key_t *, device_t>;

  using view_host_t = typename Kokkos::View<key_t *, typename view_t::host_mirror_space>;
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_ORCHARD_KEY_BASE_H
