// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ExtractNonGhostedArray.h
 */
#ifndef KALYPSSO_CORE_EXTRACT_NON_GHOSTED_ARRAY_H_
#define KALYPSSO_CORE_EXTRACT_NON_GHOSTED_ARRAY_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock, DataArrayGhostedBlock

namespace kalypsso
{

// ========================================================
// ========================================================
// ========================================================
/**
 * A simple helper structure to convert a DataArrayGhostedBlock into a DataArrayBlock (non-ghosted).
 */
template <size_t dim, typename device_t>
struct ExtractNonGhostedArray
{

  //! type alias for cell-centered data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayGhostedBlock_t data, int32_t ivar)
  {
    KALYPSSO_ASSERT(ivar < data.num_vars());

    const auto label = std::string("extracted array var=") + std::to_string(ivar);

    // data inner block size to allocate result
    auto res = DataArrayBlock_t(label, data.block_size(), 1, data.num_quadrants());

    const auto    nbCellsPerLeaf = res.num_cells();
    const int64_t total_num_cells = nbCellsPerLeaf * data.num_quadrants();

    Kokkos::parallel_for(
      label,
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        /// convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto i_oct = global_index / nbCellsPerLeaf;
        const auto i_cell = static_cast<int32_t>(global_index - i_oct * nbCellsPerLeaf);

        const auto & bSize = res.block_size();

        auto ijk = cellindex_to_coord<dim>(i_cell, bSize);

        // copy value from inner block to res
        res(ijk, 0, i_oct) = data(ijk, ivar, i_oct);
      });

    return res;

  } // run

}; // struct ExtractNonGhostedArray

} // namespace kalypsso

#endif // KALYPSSO_CORE_EXTRACT_NON_GHOSTED_ARRAY_H_
