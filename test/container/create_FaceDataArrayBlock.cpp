// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file create_FaceDataArrayBlock.cpp
 *
 */
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/FaceDataArrayBlock.h>
#include <kalypsso/core/FaceDataArrayBlock_utils.h>
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
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlock_t = typename FaceDataArrayBlock_t::DataArrayBlock_t;
  using ExecutionSpace = typename device_t::execution_space;

  const auto bsize = get_block_size<dim>(4);
  const auto num_octants = 10;
  const auto num_cells_per_leaf = Kokkos::dim_prod(bsize);
  const auto num_cells = num_octants * num_cells_per_leaf;

  FaceDataArrayBlock_t fdata("fdata", bsize, num_octants);

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
      if (fdata.num_quadrants() != fdata.num_quadrants())
        dummy++;
#endif

      if constexpr (dim == 2)
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];

        fdata(i, j, IX, iOct) = 1.0 * (i + j);
        fdata(i, j, IY, iOct) = 1.0 * (i - j);
        fdata(i, j, IZ, iOct) = 0.0;

        // init right faces of the last cell
        if (i == bsize[IX] - 1)
        {
          fdata(i + 1, j, IX, iOct) = 1.0 * (i + 1 + j);
        }
        if (j == bsize[IY] - 1)
        {
          fdata(i, j + 1, IY, iOct) = 1.0 * (i - j - 1);
        }
      }
      else if constexpr (dim == 3)
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];
        auto const & k = iCoord[IZ];

        fdata(i, j, k, IX, iOct) = i + j + k;
        fdata(i, j, k, IY, iOct) = i - j + k;
        fdata(i, j, k, IZ, iOct) = i + j - k;

        // init right faces of the last cell
        if (i == bsize[IX] - 1)
        {
          fdata(i + 1, j, k, IX, iOct) = 1.0 * (i + 1 + j + k);
        }
        if (j == bsize[IY] - 1)
        {
          fdata(i, j + 1, k, IY, iOct) = 1.0 * (i - j - 1 + k);
        }
        if (k == bsize[IZ] - 1)
        {
          fdata(i, j, k + 1, IZ, iOct) = 1.0 * (i + j - k - 1);
        }
      }
    });

  auto res = DataArrayBlock_t("fdata_centered", bsize, 1, num_octants);

  const int dir = IX;

  // compute unit vector associated to direction
  const auto unit_vector = [](int direction) {
    Kokkos::Array<int, dim> v;
    for (int i = 0; i < dim; ++i)
    {
      v[i] = i == direction ? 1 : 0;
    }
    return v;
  }(dir);

  Kokkos::parallel_for(
    "convert",
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
      if ((fdata.num_quadrants() == 0) or (res.num_cells() == 0) or (dir == 0) or
          (unit_vector[0] == 0))
        dummy++;
#endif
      if constexpr (dim == 2)
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];

        res(cell_index, 0, iOct) =
          HALF_F *
          (fdata(i, j, dir, iOct) + fdata(i + unit_vector[IX], j + unit_vector[IY], dir, iOct));
      }
      else
      {
        auto const & i = iCoord[IX];
        auto const & j = iCoord[IY];
        auto const & k = iCoord[IZ];

        res(cell_index, 0, iOct) =
          HALF_F *
          (fdata(i, j, k, dir, iOct) +
           fdata(i + unit_vector[IX], j + unit_vector[IY], k + unit_vector[IZ], dir, iOct));
      }
    });

  auto fdata_centered = FaceDataArrayBlock_t::to_DataArrayBlockCentered(fdata, IX);

  const auto fdata_centered_h = DataArrayBlock_t::create_host_mirror_view_and_copy(fdata_centered);

  for (auto iOct = 0; iOct < num_octants; ++iOct)
    for (uint32_t icell = 0; icell < Kokkos::dim_prod(bsize); ++icell)
    {
      KALYPSSO_INFO_ALL("fdata_centered[{},{},{}]={}",
                        icell,
                        static_cast<int>(IX),
                        0,
                        fdata_centered_h(icell, 0, iOct));
    }

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
