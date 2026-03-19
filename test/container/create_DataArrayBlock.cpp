// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file create_DataArrayBlock.cpp
 *
 */
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/DataArrayBlock.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/utils/log/kalypsso_log.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/cmdline_utils.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>

namespace kalypsso
{

template <int dim, typename device_t>
void
test(const ParallelEnv & par_env)
{
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using ExecutionSpace = typename device_t::execution_space;

  const auto bsize = get_block_size<dim>(4);
  const auto num_vars = 2;
  const auto num_octants = 10;
  const auto num_cells_per_leaf = Kokkos::dim_prod(bsize);
  const auto num_cells = num_octants * num_cells_per_leaf;

  auto data = DataArrayBlock_t("somedata", bsize, num_vars, num_octants);

  Kokkos::parallel_for(
    "init",
    Kokkos::RangePolicy<ExecutionSpace>(0, num_cells),
    KOKKOS_LAMBDA(const uint32_t & global_index) {
      // convert global index into
      // - octant id
      // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
      const auto iOct = global_index / num_cells_per_leaf;
      const auto cell_index = global_index - iOct * num_cells_per_leaf;

      // compute ix,iy,iz of local cell inside
      // block from index
      const auto iCoord = cellindex_to_coord<dim>(cell_index, bsize);

#ifdef __NVCC__
      [[maybe_unused]] int dummy = 0;
      if (data.num_quadrants() != data.num_quadrants())
        dummy++;
#endif

      if constexpr (dim == 2)
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];

        data(i, j, 0, iOct) = 1.0 * i + 1.0 * j;
        data(i, j, 1, iOct) = 1.0 * i - 1.0 * j;
      }
      else if constexpr (dim == 3)
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];
        auto const & k = iCoord[IZ];

        data(i, j, k, 0, iOct) = 1.0 * i + j + k;
        data(i, j, k, 1, iOct) = 1.0 * i - j + k;
      }
    });

  const auto ivar = 1;
  const auto data_h = DataArrayBlock_t::create_host_mirror_view_and_copy(data);

#ifdef KALYPSSO_CORE_USE_SPDLOG
  if constexpr (dim == 2)
  {
    for (auto iOct = 0; iOct < num_octants; ++iOct)
    {
      for (uint32_t icell = 0; icell < Kokkos::dim_prod(bsize); ++icell)
      {
        fmt::print("data[{},{},{}]={:4}\n", icell, ivar, iOct, data_h(icell, ivar, iOct));
      }
    }

    for (auto iOct = 0; iOct < num_octants; ++iOct)
    {
      for (uint32_t j = 0; j < bsize[IY]; ++j)
      {
        for (uint32_t i = 0; i < bsize[IX]; ++i)
        {
          fmt::print("data[{},{},{},{}]={:4} ", i, j, ivar, iOct, data_h(i, j, ivar, iOct));
        }
        fmt::print("\n");
      }
    }
  }
  else if constexpr (dim == 3)
  {
    for (auto iOct = 0; iOct < 2; ++iOct)
    {
      for (uint32_t icell = 0; icell < Kokkos::dim_prod(bsize); ++icell)
      {
        fmt::print("data[{},{},{}]={:4}\n", icell, ivar, iOct, data_h(icell, ivar, iOct));
      }
    }

    for (auto iOct = 0; iOct < 2; ++iOct)
    {
      for (uint32_t k = 0; k < bsize[IZ]; ++k)
      {
        for (uint32_t j = 0; j < bsize[IY]; ++j)
        {
          for (uint32_t i = 0; i < bsize[IX]; ++i)
          {
            fmt::print(
              "data[{},{},{},{},{}]={:4} ", i, j, k, ivar, iOct, data_h(i, j, k, ivar, iOct));
          }
          fmt::print("\n");
        }
        fmt::print("\n\n");
      }
    }
  }
#endif // KALYPSSO_CORE_USE_SPDLOG
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

  if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
  {
    if (par_env.rank() == 0)
    {
      std::cout << "Example cmdline: \"./create_DataArrayBlock\"\n";
    }
    return EXIT_SUCCESS;
  }

  // parse command line : 2d or 3d ?
  bool do_3d = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

  using DefaultDevice =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

  if (do_3d)
  {
    kalypsso::test<3, DefaultDevice>(par_env);
  }
  else
  {
    kalypsso::test<2, DefaultDevice>(par_env);
  }

  return EXIT_SUCCESS;

} // main
