// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeCurvature.cpp
 */
#include <kalypsso/core/ComputeCurvature.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeCurvature<dim, device_t>::ComputeCurvature(DataArrayGhostedBlock_t normal_vector,
                                                  DataArrayGhostedBlock_t curvature,
                                                  const OrchardKeys &     keys,
                                                  const int32_t           iOct_first,
                                                  const int32_t           num_quads,
                                                  ConfigMap const &       config_map)
  : m_normal_vector(normal_vector)
  , m_curvature(curvature)
  , m_keys(keys)
  , m_iOct_first(iOct_first)
  , m_num_quads(num_quads)
  , m_scaling_factor(get_scaling_factor(config_map))

{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeCurvature<dim, device_t>::apply(DataArrayGhostedBlock_t normal_vector,
                                       DataArrayGhostedBlock_t curvature,
                                       const OrchardKeys &     keys,
                                       const int32_t           iOct_first,
                                       const int32_t           num_quads,
                                       ConfigMap const &       config_map)
{

  ComputeCurvature<dim, device_t> functor(
    normal_vector, curvature, keys, iOct_first, num_quads, config_map);

  const auto nbIterations = num_quads * curvature.num_cells();

  // launch computation
  Kokkos::parallel_for(
    "kalypsso::core::ComputeCurvature", Kokkos::RangePolicy<exec_space>(0, nbIterations), functor);

} // ComputeCurvature<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeCurvature<dim, device_t>::operator()(const index_t & global_index) const
{
  const auto block_size = m_curvature.block_size();

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const i_oct = static_cast<int32_t>(global_index / m_curvature.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - i_oct * m_curvature.num_cells());

  // compute cell length
  const auto level = orchard_key_t<dim>::level(m_keys(i_oct));
  const auto dx = compute_cell_length<dim>(level, block_size[IX]) * m_scaling_factor;


  // compute cartesian coordinates inside ghosted block
  const auto coord =
    cellindex_to_coord<dim>(cell_index, m_curvature.ghosted_block_size(), m_curvature.shift());

  if constexpr (dim == 2)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    // clang-format off
    const auto dn_dx =
      (m_normal_vector(i + 1, j, IX, i_oct) -
       m_normal_vector(i - 1, j, IX, i_oct)) / 2;
    const auto dn_dy =
      (m_normal_vector(i, j + 1, IY, i_oct) -
       m_normal_vector(i, j - 1, IY, i_oct)) / 2;
    // clang-format on

    // compute divergence
    m_curvature(i, j, 0, i_oct) = -(dn_dx + dn_dy) / dx;
  }
  else if constexpr (dim == 3)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    // clang-format off
    const auto dn_dx =
      (m_normal_vector(i + 1, j, k, IX, i_oct) -
       m_normal_vector(i - 1, j, k, IX, i_oct)) / 2;
    const auto dn_dy =
      (m_normal_vector(i, j + 1, k, IY, i_oct) -
       m_normal_vector(i, j - 1, k, IY, i_oct)) / 2;
    const auto dn_dz =
      (m_normal_vector(i, j, k + 1, IZ, i_oct) -
       m_normal_vector(i, j, k - 1, IZ, i_oct)) / 2;
    // clang-format on

    // compute divergence
    m_curvature(i, j, k, 0, i_oct) = -(dn_dx + dn_dy + dn_dz) / dx;
  }

} // ComputeCurvature<dim, device_t>::operator ()

// explicit template instantiation
template class ComputeCurvature<2, kalypsso::DefaultDevice>;
template class ComputeCurvature<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
