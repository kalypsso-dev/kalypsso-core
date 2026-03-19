// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SmoothInterfaceFunctionData.cpp
 * \brief Definition of a container for data needed in smooth interface function
 * related algorithms.
 *
 * Algorithms include:
 *
 * - THINC: Tangent Hyperbolic INterface Capturing
 * - surface tension source term as implemented in Garrick et al (2017)
 *
 * References:
 *
 * - A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension,
 * Garrick Owkes and Regele, Journal of Computational Physics Volume 339, 15 June 2017, Pages 46-67.
 * https://doi.org/10.1016/j.jcp.2017.03.007
 * - An interface capturing scheme for modeling atomization in compressible flows, Garrick, Hagen
 * and Regele, Journal of Computational Physics, Volume 344, 1 September 2017, Pages 260-280.
 * https://doi.org/10.1016/j.jcp.2017.04.079
 *
 */

#include <kalypsso/core/SmoothInterfaceFunctionData.h>

#include <kalypsso/core/ComputeSmoothInterfaceFunction.h>
#include <kalypsso/core/FirstOrderDerivativeFiniteDifference.h>
#include <kalypsso/core/FillBlockGhostCellsInPlace.h>
#include <kalypsso/core/ComputeFilteredCurvature.h>

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
SmoothInterfaceFunctionData<dim, device_t>::resize()
{

  const auto num_quadrants = m_mesh_map.get_amr_mesh_info().local_num_quadrants_total();

  if (m_enabled)
  {
    m_sif.resize(num_quadrants);
    m_normal_vector.resize(num_quadrants);

    if (m_surface_tension_enabled)
    {
      m_curvature_weights.resize(num_quadrants);
      m_unfiltered_curvature.resize(num_quadrants);
      m_curvature.resize(num_quadrants);
    }
  }

} // SmoothInterfaceFunctionData<dim, device_t>::resize

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
uint64_t
SmoothInterfaceFunctionData<dim, device_t>::total_mem_size_in_bytes() const
{
  uint64_t total = 0;

  if (m_enabled)
  {
    // the following array are needed for THINC computation
    total += m_sif.allocated_size_in_bytes();
    total += m_normal_vector.allocated_size_in_bytes();

    // the following arrays are only need for surface tension computation
    if (m_surface_tension_enabled)
    {
      total += m_curvature_weights.allocated_size_in_bytes();
      total += m_unfiltered_curvature.allocated_size_in_bytes();
      total += m_curvature.allocated_size_in_bytes();
    }
  }

#ifdef KALYPSSO_CORE_USE_MPI
  total += m_mesh_ghosts_exchanger.allocated_size_in_bytes();
#endif

  return total;

} // SmoothInterfaceFunctionData<dim, device_t>::total_mem_size_in_bytes

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
SmoothInterfaceFunctionData<dim, device_t>::compute_smooth_interface_function(
  DataArrayBlock_t const & userdata_in,
  int32_t                  ivar)
{
  KOKKOS_ASSERT(userdata_in.num_quadrants() == m_sif.num_quadrants() && "Wrong sizes");

  // fill m_sif
  ComputeSmoothInterfaceFunction<dim, device_t>::apply(
    m_config_map, userdata_in, m_sif, ivar, 0, m_sif.num_quadrants());

} // SmoothInterfaceFunctionData<dim, device_t>::compute_smooth_interface_function

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
SmoothInterfaceFunctionData<dim, device_t>::compute_interface_normal_vector(
  const OrchardKeys & keys)
{
  const auto stencil_length = core::FIRST_DERIVATIVE_STENCIL::SEVEN_POINTS;

  // compute normal vector
  core::FirstOrderDerivativeFiniteDifference<dim, device_t>::normalized_gradient(
    m_config_map, keys, 0, m_sif.num_quadrants(), m_sif, 0, m_normal_vector, stencil_length);

  // fill ghost cells around block
  FillBlockGhostCellsInPlaceFunctor<dim, device_t>::apply(m_config_map,
                                                          m_mesh_map.hashmap(),
                                                          keys,
                                                          m_normal_vector,
                                                          get_brick_sizes<dim>(m_config_map),
                                                          get_brick_periodicity<dim>(m_config_map));

#ifdef KALYPSSO_CORE_USE_MPI
  m_mesh_ghosts_exchanger.exchange(m_normal_vector.data());
#endif

} // SmoothInterfaceFunctionData<dim, device_t>::compute_interface_normal_vector

// =========================================================================
// =========================================================================
template <size_t dim, typename device_t>
void
SmoothInterfaceFunctionData<dim, device_t>::compute_curvature(const OrchardKeys &             keys,
                                                              DataArrayGhostedBlock_t const & qdata,
                                                              int32_t                         iphi)
{

  if (m_surface_tension_enabled)
  {
    const auto stencil_length = core::FIRST_DERIVATIVE_STENCIL::SEVEN_POINTS;

    core::FirstOrderDerivativeFiniteDifference<dim, device_t>::divergence(
      m_config_map,
      keys,
      0,
      m_normal_vector.num_quadrants(),
      m_normal_vector,
      m_unfiltered_curvature,
      stencil_length,
      -1.0);

    // fill ghost cells around block
    FillBlockGhostCellsInPlaceFunctor<dim, device_t>::apply(
      m_config_map,
      m_mesh_map.hashmap(),
      keys,
      m_unfiltered_curvature,
      get_brick_sizes<dim>(m_config_map),
      get_brick_periodicity<dim>(m_config_map));

#ifdef KALYPSSO_CORE_USE_MPI
    m_mesh_ghosts_exchanger.exchange(m_unfiltered_curvature.data());
#endif

    Kokkos::deep_copy(m_curvature.data().logical_view(),
                      m_unfiltered_curvature.data().logical_view());

    const auto num_filt_iter =
      m_config_map.getInteger("smooth_interface_function", "curvature_filter_iterations", 3);

    if (num_filt_iter > 0)
    {
      // apply filter
      ComputeFilteredCurvature<dim, device_t>::apply(qdata,
                                                     iphi,
                                                     m_unfiltered_curvature,
                                                     m_curvature_weights,
                                                     m_curvature,
                                                     0,
                                                     m_curvature.num_quadrants(),
                                                     m_config_map);


      // if the number of filtering steps is even, we need to swap array so stat the final filtered
      // curvature is contained in m_curvature
      if (num_filt_iter % 2 == 0)
      {
        my_swap(m_curvature, m_unfiltered_curvature);
      }

#ifdef KALYPSSO_CORE_USE_MPI
      m_mesh_ghosts_exchanger.exchange(m_curvature.data());
#endif
    }
    else
    {
      Kokkos::deep_copy(m_curvature.data().logical_view(),
                        m_unfiltered_curvature.data().logical_view());
    }
  }
  else
  {
    KALYPSSO_ERROR(
      "Cannot compute curvature; please set \"smooth_interface_function/surface_tension_enabled\" ",
      "to true in the input parameter file.");
  }

} // SmoothInterfaceFunctionData<dim, device_t>::compute_curvature

// explicit template instantiation
template class SmoothInterfaceFunctionData<2, kalypsso::DefaultDevice>;
template class SmoothInterfaceFunctionData<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
