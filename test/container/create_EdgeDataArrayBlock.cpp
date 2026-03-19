// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file create_EdgeDataArrayBlock.cpp
 *
 */
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/EdgeDataArrayBlock.h>
#include <kalypsso/core/EdgeDataArrayBlock_utils.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>
#include <cstdint>

namespace kalypsso
{

template <int dim, typename device_t>
void
test(const ParallelEnv & par_env)
{
  using EdgeDataArrayBlock_t = EdgeDataArrayBlock<dim, real_t, device_t>;
  // using DataArrayBlock_t = typename EdgeDataArrayBlock_t::DataArrayBlock_t;
  using ExecutionSpace = typename device_t::execution_space;

  const auto bsize = get_block_size<dim>(4);
  const auto num_octants = 10;

  EdgeDataArrayBlock_t edata("edata", bsize, num_octants);

  const auto num_elements_per_octant = edata.num_elements_per_octant();

  Kokkos::parallel_for(
    "init",
    Kokkos::RangePolicy<ExecutionSpace>(0, num_elements_per_octant * num_octants),
    KOKKOS_LAMBDA(const uint32_t & global_index) {
      // convert global index into
      // - octant id
      // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
      const auto iOct = global_index / num_elements_per_octant;
      const auto edge_index = global_index - iOct * num_elements_per_octant;

      // compute ix,iy,iz of current edge
      auto const edge_indexes = edge_flat_index_unravel<dim>(edge_index, bsize, edata.offsets());

      if constexpr (dim == 2)
      {
        auto const & i = edge_indexes[IX];
        auto const & j = edge_indexes[IY];
        auto const & ivar = edge_indexes[dim];

        edata(i, j, ivar, iOct) = 1.0 * (i + j) + 2 * (ivar + 1);
      }
      else if constexpr (dim == 3)
      {
        auto const & i = edge_indexes[IX];
        auto const & j = edge_indexes[IY];
        auto const & k = edge_indexes[IZ];
        auto const & ivar = edge_indexes[dim];

        edata(i, j, k, ivar, iOct) = i + j + k + (3 * ivar + 1);
      }
    });

  // auto curl_of_edata = EdgeDataArrayBlock_t::compute_curl(edata, orchard_key);

} // test

} // namespace kalypsso

// ================================================================================
// ================================================================================
// ================================================================================
int
main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[])
{

  kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
  kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

  using DefaultDevice =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

  kalypsso::test<2, DefaultDevice>(par_env);

  return EXIT_SUCCESS;
}
