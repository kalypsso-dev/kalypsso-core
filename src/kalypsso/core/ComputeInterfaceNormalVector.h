// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeInterfaceNormalVector.h
 */
#ifndef KALYPSSO_CORE_COMPUTE_INTERFACE_NORMAL_VECTOR_H_
#define KALYPSSO_CORE_COMPUTE_INTERFACE_NORMAL_VECTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute interface normal vector as the gradient of psi, the smooth interface function using a
 * fourth order central difference scheme.
 *
 * References:
 *
 * - A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension,
 * Garrick Owkes and Regele, Journal of Computational Physics Volume 339, 15 June 2017, Pages 46-67.
 * https://doi.org/10.1016/j.jcp.2017.03.007
 * - An interface capturing scheme for modeling atomization in compressible flows, Garrick, Hagen
 * and Regele, Journal of Computational Physics, Volume 344, 1 September 2017, Pages 260-280.
 * https://doi.org/10.1016/j.jcp.2017.04.079
 */
template <size_t dim, typename device_t>
class ComputeInterfaceNormalVector
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

private:
  //! smooth interface function
  DataArrayGhostedBlock_t m_sif;

  //! normal vector
  DataArrayGhostedBlock_t m_normal_vector;

  //! offset to first octant
  const int32_t m_iOct_first;

  //! number of quadrants to process
  const int32_t m_num_quads;

public:
  // ==============================================================
  // ==============================================================
  ComputeInterfaceNormalVector(DataArrayGhostedBlock_t smooth_interface_function,
                               DataArrayGhostedBlock_t normal_vector,
                               const int32_t           iOct_first,
                               const int32_t           num_quads);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(DataArrayGhostedBlock_t smooth_interface_function,
        DataArrayGhostedBlock_t normal_vector,
        const int32_t           iOct_first,
        const int32_t           num_quads);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeInterfaceNormalVector

// explicit template instantiation
extern template class ComputeInterfaceNormalVector<2, kalypsso::DefaultDevice>;
extern template class ComputeInterfaceNormalVector<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTE_INTERFACE_NORMAL_VECTOR_H_
