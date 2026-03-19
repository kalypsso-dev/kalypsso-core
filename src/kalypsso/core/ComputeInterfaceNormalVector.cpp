// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeInterfaceNormalVector.cpp
 */
#include <kalypsso/core/ComputeInterfaceNormalVector.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeInterfaceNormalVector<dim, device_t>::ComputeInterfaceNormalVector(
  DataArrayGhostedBlock_t smooth_interface_function,
  DataArrayGhostedBlock_t normal_vector,
  const int32_t           iOct_first,
  const int32_t           num_quads)
  : m_sif(smooth_interface_function)
  , m_normal_vector(normal_vector)
  , m_iOct_first(iOct_first)
  , m_num_quads(num_quads)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeInterfaceNormalVector<dim, device_t>::apply(
  DataArrayGhostedBlock_t smooth_interface_function,
  DataArrayGhostedBlock_t normal_vector,
  const int32_t           iOct_first,
  const int32_t           num_quads)
{

  ComputeInterfaceNormalVector<dim, device_t> functor(
    smooth_interface_function, normal_vector, iOct_first, num_quads);

  const auto nbIterations = num_quads * normal_vector.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::core::ComputeInterfaceNormalVector",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // ComputeInterfaceNormalVector<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeInterfaceNormalVector<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_normal_vector.num_cells());
  auto const cell_index =
    static_cast<int32_t>(global_index - iOct_local * m_normal_vector.num_cells());

  // compute cartesian coordinates inside ghosted block
  const auto coord = cellindex_to_coord<dim>(
    cell_index, m_normal_vector.ghosted_block_size(), m_normal_vector.shift());

  if constexpr (dim == 2)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];

    // clang-format off
    const auto dsif_dx = (      m_sif(i - 2, j, 0, iOct_local)
                          - 8 * m_sif(i - 1, j, 0, iOct_local)
                          + 8 * m_sif(i + 1, j, 0, iOct_local)
                          -     m_sif(i + 2, j, 0, iOct_local)) / 12;
    const auto dsif_dy = (      m_sif(i, j - 2, 0, iOct_local)
                          - 8 * m_sif(i, j - 1, 0, iOct_local)
                          + 8 * m_sif(i, j + 1, 0, iOct_local)
                          -     m_sif(i, j + 2, 0, iOct_local)) / 12;
    // clang-format on

    auto norm = sqrt(dsif_dx * dsif_dx + dsif_dy * dsif_dy);

    // if norm is very small, it means the sif is almost uniform
    // avoid division by zero
    if (norm < KALYPSSO_NUM(1e-13))
      norm = 1.0;

    m_normal_vector(i, j, IX, iOct_local) = dsif_dx / norm;
    m_normal_vector(i, j, IY, iOct_local) = dsif_dy / norm;
  }
  else if constexpr (dim == 3)
  {
    auto const & i = coord[IX];
    auto const & j = coord[IY];
    auto const & k = coord[IZ];

    // clang-format off
    const auto dsif_dx = (      m_sif(i - 2, j    , k    , 0, iOct_local)
                          - 8 * m_sif(i - 1, j    , k    , 0, iOct_local)
                          + 8 * m_sif(i + 1, j    , k    , 0, iOct_local)
                          -     m_sif(i + 2, j    , k    , 0, iOct_local)) / 12;
    const auto dsif_dy = (      m_sif(i    , j - 2, k    , 0, iOct_local)
                          - 8 * m_sif(i    , j - 1, k    , 0, iOct_local)
                          + 8 * m_sif(i    , j + 1, k    , 0, iOct_local)
                          -     m_sif(i    , j + 2, k    , 0, iOct_local)) / 12;
    const auto dsif_dz = (      m_sif(i    , j    , k - 2, 0, iOct_local)
                          - 8 * m_sif(i    , j    , k - 1, 0, iOct_local)
                          + 8 * m_sif(i    , j    , k + 1, 0, iOct_local)
                          -     m_sif(i    , j    , k + 2, 0, iOct_local)) / 12;
    // clang-format on

    auto norm = sqrt(dsif_dx * dsif_dx + dsif_dy * dsif_dy + dsif_dz * dsif_dz);

    // if norm is very small, it means the sif is almost uniform
    // avoid division by zero
    if (norm < KALYPSSO_NUM(1e-13))
      norm = 1.0;

    m_normal_vector(i, j, k, IX, iOct_local) = dsif_dx / norm;
    m_normal_vector(i, j, k, IY, iOct_local) = dsif_dy / norm;
    m_normal_vector(i, j, k, IZ, iOct_local) = dsif_dz / norm;
  }

} // ComputeInterfaceNormalVector<dim, device_t>::operator ()

// explicit template instantiation
template class ComputeInterfaceNormalVector<2, kalypsso::DefaultDevice>;
template class ComputeInterfaceNormalVector<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
