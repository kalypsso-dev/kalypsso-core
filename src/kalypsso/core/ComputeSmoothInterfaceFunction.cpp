// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeSmoothInterfaceFunction.cpp
 */
#include <kalypsso/core/ComputeSmoothInterfaceFunction.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeSmoothInterfaceFunction<dim, device_t>::ComputeSmoothInterfaceFunction(
  DataArrayBlock_t        userdata,
  DataArrayGhostedBlock_t smooth_interface_function,
  int32_t                 ivar,
  const int32_t           iOct_first,
  const int32_t           num_quads,
  const real_t            alpha)
  : m_U(userdata)
  , m_sif(smooth_interface_function)
  , m_ivar(ivar)
  , m_iOct_first(iOct_first)
  , m_num_quads(num_quads)
  , m_alpha(alpha)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeSmoothInterfaceFunction<dim, device_t>::apply(
  ConfigMap const &       config_map,
  DataArrayBlock_t        userdata,
  DataArrayGhostedBlock_t smooth_interface_function,
  const int32_t           ivar,
  const int32_t           iOct_first,
  const int32_t           num_quads)
{

  ComputeSmoothInterfaceFunction<dim, device_t> functor(
    userdata,
    smooth_interface_function,
    ivar,
    iOct_first,
    num_quads,
    config_map.getReal("smooth_interface_function", "alpha", KALYPSSO_NUM(0.1)));

  const auto nbIterations = num_quads * userdata.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::core::ComputeSmoothInterfaceFunction",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeSmoothInterfaceFunction<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_U.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_U.num_cells());

  KOKKOS_ASSERT(iOct_local < m_U.num_quadrants());
  KOKKOS_ASSERT(iOct_local < m_sif.num_quadrants());

  const auto phi = m_U(cell_index, m_ivar, iOct_local);

  // we use ijk here because m_sif is a ghosted array
  const auto ijk = cellindex_to_coord<dim>(cell_index, m_U.block_size());

  m_sif(ijk, 0, iOct_local) = pow(phi, m_alpha) / (pow(phi, m_alpha) + pow(ONE_F - phi, m_alpha));

} // operator ()

// explicit template instantiation
template class ComputeSmoothInterfaceFunction<2, kalypsso::DefaultDevice>;
template class ComputeSmoothInterfaceFunction<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
