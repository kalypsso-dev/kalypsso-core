// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file LinearCombination.h
 */
#ifndef KALYPSSO_CORE_LINEARCOMBINATION_H_
#define KALYPSSO_CORE_LINEARCOMBINATION_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/utils_block.h>

namespace kalypsso
{

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class LinearCombination
 *
 * \tparam dim dimension
 * \tparam device_t Kokkos device
 *
 * This a helper class for computing linear combination of two DataArrayBlock's of same sizes; the
 * result is put in the second array.
 *
 */
template <size_t dim, typename device_t>
class LinearCombination
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  /**
   * Computes linear combination: data1 = coefs[0]*data0 + coefs[1]*data1
   */
  static void
  apply(DataArrayBlock_t         data0,
        DataArrayBlock_t         data1,
        Kokkos::Array<real_t, 2> coefs,
        int32_t                  num_octants)
  {
    KOKKOS_ASSERT((data0.num_cells() == data1.num_cells()) &&
                  "data0 and data1 must have same number of cells per quadrant");
    KOKKOS_ASSERT((data0.num_vars() == data1.num_vars()) &&
                  "data0 and data1 must have same number of variables");
    KOKKOS_ASSERT((data0.num_quadrants() == data1.num_quadrants()) &&
                  "data0 and data1 must have same number of quadrants");
    KOKKOS_ASSERT((num_octants <= data0.num_quadrants()) && "Wrong number of octants");

    const auto num_cells = data0.num_cells();
    const auto total_num_cells = num_cells * num_octants;
    const auto num_vars = data0.num_vars();

    Kokkos::parallel_for(
      "LinearCombination",
      Kokkos::RangePolicy<exec_space>(0, total_num_cells),
      KOKKOS_LAMBDA(const int32_t & global_index) {
        const auto iOct = global_index / num_cells;
        const auto cell_index = global_index - iOct * num_cells;

        for (int32_t ivar = 0; ivar < num_vars; ++ivar)
          data1(cell_index, ivar, iOct) =
            coefs[0] * data0(cell_index, ivar, iOct) + coefs[1] * data1(cell_index, ivar, iOct);
      });

  } // apply

  static void
  apply(FaceDataArrayBlock_t     data0,
        FaceDataArrayBlock_t     data1,
        Kokkos::Array<real_t, 2> coefs,
        int32_t                  num_octants)
  {

    KOKKOS_ASSERT((data0.num_elements_per_octant() == data1.num_elements_per_octant()) &&
                  "data0 and data1 must have same number of elements per octant");

    const auto nbFacePerLeaf = data0.num_elements_per_octant();
    const auto total_num_faces = nbFacePerLeaf * num_octants;

    Kokkos::parallel_for(
      "LinearCombination",
      Kokkos::RangePolicy<exec_space>(0, total_num_faces),
      KOKKOS_LAMBDA(const int32_t & global_index) {
        const auto    iOct = global_index / nbFacePerLeaf;
        const int32_t face_flat_index = static_cast<int32_t>(global_index - iOct * nbFacePerLeaf);
        const auto &  block_sizes = data0.cell_block_size();
        const auto    face_indexes = face_flat_index_unravel<dim>(
          face_flat_index, block_sizes, data0.offsets(), data0.shift());

        data1(face_indexes, iOct) =
          coefs[0] * data0(face_indexes, iOct) + coefs[1] * data1(face_indexes, iOct);
      });

  } // apply
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_LINEARCOMBINATION_H_
