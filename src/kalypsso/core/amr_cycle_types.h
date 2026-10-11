// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file amr_cycle_types.h
 *
 * \brief Defines useful type aliases for implementing AMR cycle (AMR manager).
 */
#ifndef KALYPSSO_CORE_AMR_CYCLE_TYPES_H_
#define KALYPSSO_CORE_AMR_CYCLE_TYPES_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/enums.h>
#include <kalypsso/core/kalypsso_core_config.h>

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRContext.h>

namespace kalypsso
{

namespace core
{

//! Array of Orchard keys
template <typename device_t>
using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

//! Hashmap type from Orchard keys to octant index
template <typename device_t>
using AMRHashmap = typename hashmap_base_t<device_t>::map_t;

//! AMR flags array type
template <size_t dim, typename device_t>
using AMRFlags = typename AMRContext<dim, device_t>::amrflags_view_t;


} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMR_CYCLE_TYPES_H_
