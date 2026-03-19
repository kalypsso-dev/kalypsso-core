// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file brick_base.h
 */
#ifndef KALYPSSO_CORE_BRICKBASE_H_
#define KALYPSSO_CORE_BRICKBASE_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>

#include <cstdint>

namespace kalypsso
{
//! A type alias to use uniformly in kalypsso for holding brick sizes
template <size_t dim>
using brick_size_t = Kokkos::Array<int8_t, dim>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_BRICKBASE_H_
