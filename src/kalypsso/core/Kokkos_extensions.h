// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Kokkos_extensions.h
 * Define some extensions to kokkos.
 *
 * Adapted from Arborx (BSD-3-Clause).
 * https://github.com/arborx/ArborX
 */
#ifndef KALYPSSO_CORE_KOKKOSEXTENSIONS_H_
#define KALYPSSO_CORE_KOKKOSEXTENSIONS_H_

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <initializer_list>
#include <type_traits>

namespace KokkosExt
{

// !Compute the maximum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &
max(T const & a, T const & b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &
min(T const & a, T const & b)
{
  return (a < b) ? a : b;
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T
max(std::initializer_list<T> ilist)
{
  auto const *       first = ilist.begin();
  auto const * const last = ilist.end();
  auto               result = *first;
  if (first == last)
  {
    return result;
  }
  while (++first != last)
  {
    if (result < *first)
    {
      result = *first;
    }
  }
  return result;
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T
min(std::initializer_list<T> ilist)
{
  auto const *       first = ilist.begin();
  auto const * const last = ilist.end();
  auto               result = *first;
  if (first == last)
  {
    return result;
  }
  while (++first != last)
  {
    if (*first < result)
    {
      result = *first;
    }
  }
  return result;
}

//! Compute the maximum of two values (passed by value).
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const
max_val(T const a, T const b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values (passed by value).
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const
min_val(T const a, T const b)
{
  return (a < b) ? a : b;
}

template <class ExecutionSpace, class View>
typename View::non_const_type
clone(ExecutionSpace const & space, View const & v, std::string const & label)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<View>::value);
  typename View::non_const_type w(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label),
                                  v.layout());
  Kokkos::deep_copy(space, w, v);
  return w;
}

template <class ExecutionSpace, class View>
typename View::non_const_type
clone(ExecutionSpace const & space, View const & v)
{
  return clone(space, v, v.label());
}

template <class ExecutionSpace, class View>
typename View::non_const_type
cloneWithoutInitializingNorCopying(ExecutionSpace const & space, View const & v)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<View>::value);
  return Kokkos::create_mirror(
    Kokkos::view_alloc(typename View::memory_space{}, space, Kokkos::WithoutInitializing), v);
}

//! clone a Kokkos::UnorderedMap
//! \todo not compiling, analyze why
template <class Map>
Map
clone_unordered_map(Map const & map)
{
  Map map2;
  // note: deep_copy for unordered map is doing memory allocation
  Kokkos::deep_copy(map2, map);
  return map2;
}

template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<
  MemorySpace,
  ExecutionSpace,
  std::enable_if_t<Kokkos::SpaceAccessibility<ExecutionSpace, MemorySpace>::accessible>>
  : std::true_type
{};

template <typename View>
struct is_accessible_from_host
  : public is_accessible_from<typename View::memory_space, Kokkos::HostSpace>
{
  static_assert(Kokkos::is_view<View>::value);
};


} // namespace KokkosExt

#endif // KALYPSSO_CORE_KOKKOSEXTENSIONS_H_
