// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file kokkos_shared.h
 */
#ifndef KALYPSSO_CORE_KOKKOS_SHARED_H_
#define KALYPSSO_CORE_KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <kalypsso/core/real_type.h>
#include <kalypsso/core/misc_utils.h>

namespace kalypsso
{

//! Kokkos default device
using DefaultDevice =
  Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

//! Kokkos host device
using HostDevice = Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;

// =============================================================
// =============================================================
/**
 * a dummy swap device routine.
 */
template <class T>
KOKKOS_INLINE_FUNCTION void
my_swap(T & a, T & b)
{
  T c{ std::move(a) };
  a = std::move(b);
  b = std::move(c);
} // my_swap

// =============================================================
// =============================================================
/**
 * Convert a std::vector in a host Kokkos::View
 */
template <typename T>
Kokkos::View<T *, Kokkos::HostSpace>
to_host_view(std::vector<T> const & vec)
{
  const auto n = vec.size();
  const auto data = vec.data();

  Kokkos::View<T *, Kokkos::HostSpace> res(
    Kokkos::view_alloc(Kokkos::HostSpace{}, Kokkos::WithoutInitializing, ""), n);

  Kokkos::parallel_for(
    "copy vector data",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
    KOKKOS_LAMBDA(size_t i) { res(i) = data[i]; });

  return res;
}

// =============================================================
// =============================================================
/**
 * Convert a std::vector in a device Kokkos::View
 */
template <typename T, typename device_t>
Kokkos::View<T *, typename device_t::memory_space>
to_view(std::vector<T> const & vec)
{
  auto res_h = to_host_view(vec);

  return Kokkos::create_mirror_view_and_copy(typename device_t::memory_space{}, res_h);
}

} // namespace kalypsso

// adapted from Kokkos
#if !defined(__NVCC__) && (!defined(KOKKOS_COMPILER_INTEL) || KOKKOS_COMPILER_INTEL >= 2021)
#  define KALYPSSO_DEPRECATED [[deprecated]]
#  define KALYPSSO_DEPRECATED_WITH_COMMENT(comment) [[deprecated(comment)]]
#else
#  define KALYPSSO_DEPRECATED
#  define KALYPSSO_DEPRECATED_WITH_COMMENT(comment)
#endif

#endif // KALYPSSO_CORE_KOKKOS_SHARED_H_
