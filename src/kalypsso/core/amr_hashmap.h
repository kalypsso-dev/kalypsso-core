// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file amr_hashmap.h
 */
#ifndef KALYPSSO_CORE_AMRHASHMAP_H
#define KALYPSSO_CORE_AMRHASHMAP_H

#include <kalypsso/core/kokkos_shared.h>
#include <Kokkos_UnorderedMap.hpp>

#include <kalypsso/core/orchard_key_base.h>

namespace kalypsso
{

// ========================================================================================
// ========================================================================================
template <typename device_t>
struct hashmap_base_t
{
  using value_t = iOct_t;

  //! type alias for a device hashmap with key=orchard_key and value=memory index
  using map_t = Kokkos::UnorderedMap<key_t, value_t, device_t>;

}; // struct hashmap_base_t

} // namespace kalypsso

#endif //  KALYPSSO_CORE_AMRHASHMAP_H
