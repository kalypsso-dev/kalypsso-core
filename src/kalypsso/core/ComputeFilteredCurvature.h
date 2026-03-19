// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFilteredCurvature.h
 */
#ifndef KALYPSSO_CORE_COMPUTE_FILTERED_CURVATURE_H_
#define KALYPSSO_CORE_COMPUTE_FILTERED_CURVATURE_H_

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
 * Compute interface curvature as the divergence of the normal vector using a
 * second order central difference scheme.
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
class ComputeFilteredCurvature
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  struct TagComputeWeights
  {};
  struct TagApplyFilter
  {};

private:
  //! Conservative variable (to access volume fractions)
  DataArrayGhostedBlock_t m_q;

  //! index to volume fraction
  int32_t m_iphi;

  //! Unfiltered curvature
  DataArrayGhostedBlock_t m_unfiltered_curvature;

  //! Curvature weights
  DataArrayGhostedBlock_t m_weights;

  //! Filtered curvature
  DataArrayGhostedBlock_t m_curvature;

  //! offset to first octant
  const int32_t m_iOct_first;

  //! Number of quadrants to process
  const int32_t m_num_quads;

  //! Number of filter iterations
  const int32_t m_num_filt_iter;

  //! filter threshold in unit of phi^2*(1-phi)^2
  const real_t m_filt_threshold;

  KALYPSSO_STATIC_MATH_CONSTANT(ONE_EM6, 1e-6);

public:
  // ==============================================================
  // ==============================================================
  ComputeFilteredCurvature(DataArrayGhostedBlock_t const & qdata,
                           const int32_t                   iphi_id,
                           DataArrayGhostedBlock_t const & unfiltered_curvature,
                           DataArrayGhostedBlock_t const & curvature_weights,
                           DataArrayGhostedBlock_t const & curvature,
                           const int32_t                   iOct_first,
                           const int32_t                   num_quads,
                           ConfigMap const &               config_map);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(DataArrayGhostedBlock_t const & qdata,
        const int32_t                   iphi_id,
        DataArrayGhostedBlock_t const & unfiltered_curvature,
        DataArrayGhostedBlock_t const & curvature_weights,
        DataArrayGhostedBlock_t const & curvature,
        const int32_t                   iOct_first,
        const int32_t                   num_quads,
        ConfigMap const &               config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeWeights const &, const index_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagApplyFilter const &, const index_t & global_index) const;

}; // ComputeFilteredCurvature

// explicit template instantiation
extern template class ComputeFilteredCurvature<2, kalypsso::DefaultDevice>;
extern template class ComputeFilteredCurvature<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTE_FILTERED_CURVATURE_H_
