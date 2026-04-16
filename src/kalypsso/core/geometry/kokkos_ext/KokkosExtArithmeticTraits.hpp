// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef KALYPSSO_CORE_GEOMETRY_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP
#define KALYPSSO_CORE_GEOMETRY_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP

#include <Kokkos_NumericTraits.hpp>

namespace kalypsso::Details::KokkosExt::ArithmeticTraits
{

template <class T>
using infinity = Kokkos::Experimental::infinity<T>;

template <class T>
using finite_max = Kokkos::Experimental::finite_max<T>;

template <class T>
using finite_min = Kokkos::Experimental::finite_min<T>;

template <class T>
using epsilon = Kokkos::Experimental::epsilon<T>;

} // namespace kalypsso::Details::KokkosExt::ArithmeticTraits

#endif // KALYPSSO_CORE_GEOMETRY_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP
