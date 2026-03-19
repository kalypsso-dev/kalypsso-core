// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file myKokkos_Sort.h
 * A custom variation for 2D bucket sort (not available right now in Kokkos).
 */

#ifndef KALYPSSO_CORE_MY_KOKKOS_SORT_HPP_
#define KALYPSSO_CORE_MY_KOKKOS_SORT_HPP_

#include <Kokkos_Core.hpp>

#include <algorithm>

// =====================================================================
// =====================================================================
// =====================================================================
namespace Kokkos
{

template <class KeyViewType>
struct BinOp2D
{
  int                                        max_bins_[2];
  double                                     mul_[2];
  typename KeyViewType::non_const_value_type range_[2];
  typename KeyViewType::non_const_value_type min_[2];

  BinOp2D() {}

  BinOp2D(int                                    max_bins__[],
          typename KeyViewType::const_value_type min[],
          typename KeyViewType::const_value_type max[])
  {
    max_bins_[0] = max_bins__[0];
    max_bins_[1] = max_bins__[1];
    mul_[0] = 1.0 * max_bins__[0] / (max[0] - min[0]);
    mul_[1] = 1.0 * max_bins__[1] / (max[1] - min[1]);
    range_[0] = max[0] - min[0];
    range_[1] = max[1] - min[1];
    min_[0] = min[0];
    min_[1] = min[1];
  }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION int
  bin(ViewType & keys, const int & i) const
  {
    return int(((int(mul_[0] * (keys(i, 0) - min_[0])) * max_bins_[1]) +
                int(mul_[1] * (keys(i, 1) - min_[1]))));
  }

  KOKKOS_INLINE_FUNCTION
  int
  max_bins() const
  {
    return max_bins_[0] * max_bins_[1];
  }

  template <class ViewType, typename iType1, typename iType2>
  KOKKOS_INLINE_FUNCTION bool
  operator()(ViewType & keys, iType1 & i1, iType2 & i2) const
  {
    if (keys(i1, 0) > keys(i2, 0))
      return true;
    else if (keys(i1, 0) == keys(i2, 0))
    {
      if (keys(i1, 1) > keys(i2, 1))
        return true;
    }
    return false;
  }
};

} // namespace Kokkos

#endif // KALYPSSO_CORE_MY_KOKKOS_SORT_HPP_
