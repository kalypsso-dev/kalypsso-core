// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file scan_utils.h
 */
#ifndef KALYPSSO_CORE_SCANUTILS_H_
#define KALYPSSO_CORE_SCANUTILS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/Kokkos_extensions.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

// ================================================================================
// ================================================================================
/**
 * A Kokkos functor for performing a segmented parallel scan.
 *
 * \note this is actually an exclusive scan
 *
 */
template <typename device_t, size_t dim, int num_levels>
struct ScanIndexByLevel
{
  // typedef's
  using exec_space = typename device_t::execution_space;

  using key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using res_view_t = Kokkos::View<uint64_t *, exec_space>;

  // data
  key_view_t m_keys;
  res_view_t m_indexes;

  uint8_t m_level_start;
  uint8_t m_level_max;

  ScanIndexByLevel(key_view_t keys, res_view_t indexes, uint8_t level_start, uint8_t level_max)
    : m_keys(keys)
    , m_indexes(indexes)
    , m_level_start(level_start)
    , m_level_max(level_max)
  {}

  static void
  run(key_view_t keys, res_view_t indexes, uint8_t level_start, uint8_t level_max)
  {
    auto functor =
      ScanIndexByLevel<exec_space, dim, num_levels>(keys, indexes, level_start, level_max);
    const auto num_keys = keys.size();

    Kokkos::parallel_scan(
      "ScanIndexByLevel", Kokkos::RangePolicy<exec_space>(0, num_keys), functor);
  }

  using ScanValue = Kokkos::Array<int64_t, static_cast<size_t>(num_levels)>;

  KOKKOS_INLINE_FUNCTION
  void
  init(ScanValue & update) const
  {
    for (int i = 0; i < num_levels; ++i)
    {
      update[i] = 0;
    }
  } // init

  KOKKOS_INLINE_FUNCTION
  void
  join(ScanValue & update, const ScanValue & input) const
  {
    for (int i = 0; i < num_levels; ++i)
    {
      update[i] += input[i];
    }
  } // join

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const size_t i, ScanValue & update, const bool is_final) const
  {

    const auto key = m_keys(i);
    const auto level = orchard_key_t<dim>::level(key);

    // clang-format off
    if (level >= m_level_start and level < m_level_start + num_levels and level <= m_level_max)
    {
      if (is_final)
      {
        m_indexes(i) = static_cast<uint64_t>(update[level - m_level_start]);
      }
      update[level - m_level_start]++;
    }
    // clang-format on

  } // operator()

}; // struct ScanIndexByLevel

// ================================================================================
// ================================================================================
/**
 * Perform a segmented parallel scan to compute index by level.
 *
 * Example of application: when one want to fill a vtkNonOverlappingAMR object, the AMR blocks are
 * grouped by level, so each block must be attributed an id, going from 0 to N(l)-1, where N(l) is
 * the number of blocks at AMR level l.
 *
 * \param[in] keys is an array of orchard keys
 * \param[in] level_min is the minimum AMR level allowed
 * \param[in] level_max is the maximum AMR level allowed
 * \param[in] local_num_quadrants (number of owned quadrants) - MPI only
 * \param[in] par_env is MPI parallel environment - MPI only
 *
 * \return array of index (same size as input array), for each key in input,
 * this is the nth key at level "l", so return "n". So output is the array of all those "n".
 *
 * \note (June 8, 2024) we can't use auto as the return type here because nvcc is complaining
 */
template <typename device_t, size_t dim>
Kokkos::View<uint64_t *, device_t>
compute_index_by_level(typename orchard_key_base_t<device_t>::view_t keys,
                       uint8_t                                       level_min,
                       uint8_t                                       level_max,
                       [[maybe_unused]] int32_t                      local_num_quadrants,
                       [[maybe_unused]] const ParallelEnv &          par_env)
{
  using res_view_t = Kokkos::View<uint64_t *, device_t>;

  auto res =
    res_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Index by level"), keys.size());

  // here we chose a maximum number of levels that will be scanned at once; this number
  // is used as dimension of a Kokkos::Array data structured used during scan.
  // TODO: study if we could increase this number, and study how it impact performance when running
  // on GPU (using Kokkos::CUDA device)
  constexpr int num_levels = 5;

  // perform scan piece by piece (i.e. by group of num_levels levels).
  auto level_start = level_min;
  while (level_start < level_max)
  {
    ScanIndexByLevel<device_t, dim, num_levels>::run(keys, res, level_start, level_max);
    level_start += num_levels;
  }

#ifdef KALYPSSO_CORE_USE_MPI
  // now we need to update indexes when the number of MPI processes is larger than 1
  if (par_env.size() > 1)
  {

    // number of AMR levels
    const auto total_num_levels = level_max - level_min + 1;

    // compute number of keys per level in current MPI process (histogram)
    // number of octants per level will have to be MPI_Scan'ed to fix indexes
    using level_histo_view_t = Kokkos::View<uint64_t *, device_t>;
    auto level_histo = level_histo_view_t("level_histo", static_cast<size_t>(total_num_levels));

    using exec_space = typename device_t::execution_space;

    Kokkos::parallel_for(
      "compute_level_histogram",
      Kokkos::RangePolicy<exec_space>(0, local_num_quadrants),
      KOKKOS_LAMBDA(const int iOct) {
        // get AMR level
        const auto level = orchard_key_t<dim>::level(keys(iOct));
        KOKKOS_ASSERT(level >= level_min and level <= level_max and
                      "level wrong value: not in valid range.");

        Kokkos::atomic_add(&level_histo(level - level_min), 1);
      });

    // now we need to integrate that over all MPI process
    auto level_histo_scanned =
      level_histo_view_t("level_histo_scanned", static_cast<size_t>(total_num_levels));
    par_env.comm().MPI_Exscan<MpiComm::SUM>(
      level_histo.data(), level_histo_scanned.data(), total_num_levels);

    // now fix indexes with offsets computed just above
    Kokkos::parallel_for(
      "compute_level_histogram",
      Kokkos::RangePolicy<exec_space>(0, keys.size()),
      KOKKOS_LAMBDA(const int iOct) {
        // get AMR level
        const auto level = orchard_key_t<dim>::level(keys(iOct));
        KOKKOS_ASSERT(level >= level_min and level <= level_max and
                      "level wrong value: not in valid range.");

        res(iOct) += level_histo_scanned(level - level_min);
      });
  } // par_env.size() > 1
#endif // KALYPSSO_CORE_USE_MPI

  return res;

} // compute_index_by_level

} // namespace kalypsso

#endif // KALYPSSO_CORE_SCANUTILS_H_
