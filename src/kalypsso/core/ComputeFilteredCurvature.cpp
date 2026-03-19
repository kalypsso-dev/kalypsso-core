// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFilteredCurvature.cpp
 */
#include <kalypsso/core/ComputeFilteredCurvature.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeFilteredCurvature<dim, device_t>::ComputeFilteredCurvature(
  DataArrayGhostedBlock_t const & qdata,
  const int32_t                   iphi_id,
  DataArrayGhostedBlock_t const & unfiltered_curvature,
  DataArrayGhostedBlock_t const & curvature_weights,
  DataArrayGhostedBlock_t const & curvature,
  const int32_t                   iOct_first,
  const int32_t                   num_quads,
  ConfigMap const &               config_map)
  : m_q(qdata)
  , m_iphi(iphi_id)
  , m_unfiltered_curvature(unfiltered_curvature)
  , m_weights(curvature_weights)
  , m_curvature(curvature)
  , m_iOct_first(iOct_first)
  , m_num_quads(num_quads)
  , m_num_filt_iter(config_map.getInteger("smooth_interface_function", "filter_iterations", 3))
  , m_filt_threshold(config_map.getReal("smooth_interface_function", "filter_threshold", ONE_EM6_F))


{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeFilteredCurvature<dim, device_t>::apply(DataArrayGhostedBlock_t const & qdata,
                                               const int32_t                   iphi_id,
                                               DataArrayGhostedBlock_t const & unfiltered_curvature,
                                               DataArrayGhostedBlock_t const & curvature_weights,
                                               DataArrayGhostedBlock_t const & curvature,
                                               const int32_t                   iOct_first,
                                               const int32_t                   num_quads,
                                               ConfigMap const &               config_map)
{

  ComputeFilteredCurvature<dim, device_t> functor(qdata,
                                                  iphi_id,
                                                  unfiltered_curvature,
                                                  curvature_weights,
                                                  curvature,
                                                  iOct_first,
                                                  num_quads,
                                                  config_map);

  // compute weights everywhere
  const auto nbIterations = num_quads * curvature.num_cells();

  // apply filter only on the inner part of the block of cells
  const auto nbIterations_filter = num_quads * curvature.num_cells_inner();

  // first compute weights
  Kokkos::parallel_for("kalypsso::core::ComputeFilteredCurvature - compute weights",
                       Kokkos::RangePolicy<exec_space, TagComputeWeights>(0, nbIterations),
                       functor);

  auto const & num_filt_iter = functor.m_num_filt_iter;

  // apply filter
  for (int iter = 0; iter < num_filt_iter; iter++)
  {
    Kokkos::parallel_for("kalypsso::core::ComputeFilteredCurvature - apply filter",
                         Kokkos::RangePolicy<exec_space, TagApplyFilter>(0, nbIterations_filter),
                         functor);

    // swap array to get prepared for the next iteration
    if (iter < num_filt_iter - 1)
    {
      my_swap(functor.m_unfiltered_curvature, functor.m_curvature);
    }
  }

} // ComputeFilteredCurvature<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFilteredCurvature<dim, device_t>::operator()(TagComputeWeights const &,
                                                    const index_t & global_index) const
{
  // retrieve local octant index in range [0, num_quads_to_process [
  auto const i_oct = static_cast<int32_t>(global_index / m_curvature.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - i_oct * m_curvature.num_cells());

  const auto phi = m_q(cell_index, m_iphi, i_oct);

  m_weights(cell_index, 0, i_oct) = phi * phi * (ONE_F - phi) * (ONE_F - phi);
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFilteredCurvature<dim, device_t>::operator()(TagApplyFilter const &,
                                                    const index_t & global_index) const
{
  // retrieve local octant index in range [0, num_quads_to_process [
  auto const i_oct = static_cast<int32_t>(global_index / m_curvature.num_cells_inner());
  auto const cell_index =
    static_cast<int32_t>(global_index - i_oct * m_curvature.num_cells_inner());

  // compute cartesian coordinates inside ghosted block
  const auto coord = cellindex_to_coord<dim>(cell_index, m_curvature.block_size());

  if constexpr (dim == 2)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    // only apply filter where needed
    const auto w = m_weights(i, j, 0, i_oct);

    if (w > m_filt_threshold)
    {
      auto sum_w = ZERO_F;
      auto sum_w_k = ZERO_F;

      for (int dj = -1; dj < 2; ++dj)
      {
        for (int di = -1; di < 2; ++di)
        {
          const auto ii = i + di;
          const auto jj = j + dj;

          sum_w += m_weights(ii, jj, 0, i_oct);
          sum_w_k += m_weights(ii, jj, 0, i_oct) * m_unfiltered_curvature(ii, jj, 0, i_oct);
        }
      }

      m_curvature(i, j, 0, i_oct) = sum_w_k / sum_w;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    // only apply filter where needed
    const auto w = m_weights(i, j, k, 0, i_oct);

    if (w > m_filt_threshold)
    {
      auto sum_w = ZERO_F;
      auto sum_w_k = ZERO_F;

      for (int dk = -1; dk < 2; ++dk)
      {
        for (int dj = -1; dj < 2; ++dj)
        {
          for (int di = -1; di < 2; ++di)
          {
            const auto ii = i + di;
            const auto jj = j + dj;
            const auto kk = k + dk;

            sum_w += m_weights(ii, jj, kk, 0, i_oct);
            sum_w_k +=
              m_weights(ii, jj, kk, 0, i_oct) * m_unfiltered_curvature(ii, jj, kk, 0, i_oct);
          }
        }
      }

      m_curvature(i, j, k, 0, i_oct) = sum_w_k / sum_w;
    }
  }

} // ComputeFilteredCurvature<dim, device_t>::operator ()

// explicit template instantiation
template class ComputeFilteredCurvature<2, kalypsso::DefaultDevice>;
template class ComputeFilteredCurvature<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
