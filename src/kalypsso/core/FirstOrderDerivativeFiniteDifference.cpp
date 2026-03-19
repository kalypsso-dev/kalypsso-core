// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FirstOrderDerivativeFiniteDifference.cpp
 */

#include <kalypsso/core/FirstOrderDerivativeFiniteDifference.h>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
FirstOrderDerivativeFiniteDifference<dim, device_t>::FirstOrderDerivativeFiniteDifference(
  ConfigMap const &        config_map,
  orchard_key_view_t       orchard_keys,
  int32_t                  iOct_begin,
  int32_t                  num_octants,
  DataArrayGhostedBlock_t  userdata_in,
  int32_t                  ivar_in,
  DataArrayGhostedBlock_t  userdata_out,
  int32_t                  ivar_out,
  int32_t                  derivative_dir,
  FIRST_DERIVATIVE_STENCIL stencil_length,
  real_t                   scalar_factor)
  : m_keys(orchard_keys)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_userdata_in(userdata_in)
  , m_ivar_in(ivar_in)
  , m_userdata_out(userdata_out)
  , m_ivar_out(ivar_out)
  , m_derivative_dir(derivative_dir)
  , m_stencil_length(stencil_length)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_fd()
  , m_scalar_factor(scalar_factor)
{}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FirstOrderDerivativeFiniteDifference<dim, device_t>::check_args_validity(
  DataArrayGhostedBlock_t  userdata_in,
  DataArrayGhostedBlock_t  userdata_out,
  FIRST_DERIVATIVE_STENCIL stencil_length)
{
  if (userdata_in.block_size() != userdata_out.block_size())
  {
    Kokkos::abort("userdata in/out must have same sizes");
  }

  if (stencil_length == +FIRST_DERIVATIVE_STENCIL::FIVE_POINTS)
  {
    {
      const auto & b = userdata_in.block_size();
      bool         invalid_size = b[IX] < 5 or b[IY] < 5;
      if constexpr (dim == 3)
      {
        invalid_size = invalid_size or b[IZ] < 5;
      }
      if (invalid_size)
        Kokkos::abort("Userdata block size is must have at least 5 cells in direction to apply the "
                      "5 points stencil. ");
    }
  }
  if (stencil_length == +FIRST_DERIVATIVE_STENCIL::SEVEN_POINTS)
  {
    {
      const auto & b = userdata_in.block_size();
      bool         invalid_size = b[IX] < 7 or b[IY] < 7;
      if constexpr (dim == 3)
      {
        invalid_size = invalid_size or b[IZ] < 7;
      }
      if (invalid_size)
        Kokkos::abort("Userdata block size is must have at least 7 cells in direction to apply the "
                      "7 points stencil. ");
    }
  }
} // FirstOrderDerivativeFiniteDifference<dim, device_t>::check_args_validity

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FirstOrderDerivativeFiniteDifference<dim, device_t>::first_derivative(
  ConfigMap const &        config_map,
  orchard_key_view_t       orchard_keys,
  int32_t                  iOct_begin,
  int32_t                  num_octants,
  DataArrayGhostedBlock_t  userdata_in,
  int32_t                  ivar_in,
  DataArrayGhostedBlock_t  userdata_out,
  int32_t                  ivar_out,
  int32_t                  derivative_dir,
  FIRST_DERIVATIVE_STENCIL stencil_length)
{
  check_args_validity(userdata_in, userdata_out, stencil_length);

  FirstOrderDerivativeFiniteDifference<dim, device_t> functor(config_map,
                                                              orchard_keys,
                                                              iOct_begin,
                                                              num_octants,
                                                              userdata_in,
                                                              ivar_in,
                                                              userdata_out,
                                                              ivar_out,
                                                              derivative_dir,
                                                              stencil_length,
                                                              1.0 // not used
  );

  const auto nbCellsTotal = num_octants * userdata_in.num_cells_inner();

  Kokkos::parallel_for("FirstOrderDerivativeFiniteDifference - first derivative",
                       Kokkos::RangePolicy<exec_space, TagFirstDerivative>(0, nbCellsTotal),
                       functor);

}; // FirstOrderDerivativeFiniteDifference<dim, device_t>::first_derivative

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FirstOrderDerivativeFiniteDifference<dim, device_t>::normalized_gradient(
  ConfigMap const &        config_map,
  orchard_key_view_t       orchard_keys,
  int32_t                  iOct_begin,
  int32_t                  num_octants,
  DataArrayGhostedBlock_t  userdata_in,
  int32_t                  ivar_in,
  DataArrayGhostedBlock_t  userdata_out,
  FIRST_DERIVATIVE_STENCIL stencil_length)
{
  check_args_validity(userdata_in, userdata_out, stencil_length);

  if (static_cast<size_t>(userdata_out.num_vars()) < dim)
  {
    Kokkos::abort(
      "Error: userdata_out should be allocated to hold at dim components of a gradient field.");
  }

  FirstOrderDerivativeFiniteDifference<dim, device_t> functor(config_map,
                                                              orchard_keys,
                                                              iOct_begin,
                                                              num_octants,
                                                              userdata_in,
                                                              ivar_in,
                                                              userdata_out,
                                                              -1, // not used
                                                              -1, // not used
                                                              stencil_length,
                                                              1.0 // not used
  );

  const auto nbCellsTotal = num_octants * userdata_in.num_cells_inner();

  Kokkos::parallel_for("FirstOrderDerivativeFiniteDifference - normalized gradient",
                       Kokkos::RangePolicy<exec_space, TagNormalizedGradient>(0, nbCellsTotal),
                       functor);

}; // FirstOrderDerivativeFiniteDifference<dim, device_t>::normalized_gradient

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FirstOrderDerivativeFiniteDifference<dim, device_t>::divergence(
  ConfigMap const &        config_map,
  orchard_key_view_t       orchard_keys,
  int32_t                  iOct_begin,
  int32_t                  num_octants,
  DataArrayGhostedBlock_t  userdata_in,
  DataArrayGhostedBlock_t  userdata_out,
  FIRST_DERIVATIVE_STENCIL stencil_length,
  real_t                   scalar_factor)
{
  check_args_validity(userdata_in, userdata_out, stencil_length);

  if (static_cast<size_t>(userdata_in.num_vars()) < dim)
  {
    Kokkos::abort("Error: userdata_in should have at least dim components so that we can take the "
                  "divergence of it.");
  }

  FirstOrderDerivativeFiniteDifference<dim, device_t> functor(config_map,
                                                              orchard_keys,
                                                              iOct_begin,
                                                              num_octants,
                                                              userdata_in,
                                                              -1, // not used
                                                              userdata_out,
                                                              -1, // not used
                                                              -1, // not used
                                                              stencil_length,
                                                              scalar_factor);

  const auto nbCellsTotal = num_octants * userdata_in.num_cells_inner();

  Kokkos::parallel_for("FirstOrderDerivativeFiniteDifference - divergence",
                       Kokkos::RangePolicy<exec_space, TagDivergence>(0, nbCellsTotal),
                       functor);

}; // FirstOrderDerivativeFiniteDifference<dim, device_t>::divergence

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FirstOrderDerivativeFiniteDifference<dim, device_t>::operator()(TagFirstDerivative const &,
                                                                const index_t & global_index) const
{
  const auto iOct_local = global_index / m_userdata_in.num_cells_inner();
  const auto cell_index = global_index - iOct_local * m_userdata_in.num_cells_inner();

  const auto iOct_global = m_iOct_begin + iOct_local;

  const auto ijk = cellindex_to_coord<dim>(cell_index, m_userdata_in.block_size());

  real_t value = 0.0;

  if (m_derivative_dir == IX)
  {
    value = compute_first_derivative<IX>(ijk, m_ivar_in, iOct_global);
  }
  else if (m_derivative_dir == IY)
  {
    value = compute_first_derivative<IY>(ijk, m_ivar_in, iOct_global);
  }
  if constexpr (dim == 3)
  {
    if (m_derivative_dir == IZ)
    {
      value = compute_first_derivative<IZ>(ijk, m_ivar_in, iOct_global);
    }
  }

  m_userdata_out(ijk, m_ivar_out, iOct_global) = value;

} // FirstOrderDerivativeFiniteDifference<dim, device_t>::operator() - TagFirstDerivative

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FirstOrderDerivativeFiniteDifference<dim, device_t>::operator()(TagNormalizedGradient const &,
                                                                const index_t & global_index) const
{
  const auto iOct_local = global_index / m_userdata_in.num_cells_inner();
  const auto cell_index = global_index - iOct_local * m_userdata_in.num_cells_inner();

  const auto iOct_global = m_iOct_begin + iOct_local;

  const auto ijk = cellindex_to_coord<dim>(cell_index, m_userdata_in.block_size());

  const auto df_dx = compute_first_derivative<IX>(ijk, 0, iOct_global);
  const auto df_dy = compute_first_derivative<IY>(ijk, 0, iOct_global);

  real_t norm = ZERO_F;

  if constexpr (dim == 2)
  {
    norm = sqrt(df_dx * df_dx + df_dy * df_dy);

    // if norm is very small, it means the sif is almost uniform
    // avoid division by zero
    if (norm < SMALL_NORM_F)
      norm = ONE_F;

    m_userdata_out(ijk, IX, iOct_global) = df_dx / norm;
    m_userdata_out(ijk, IY, iOct_global) = df_dy / norm;
  }
  else if constexpr (dim == 3)
  {
    const auto df_dz = compute_first_derivative<IZ>(ijk, 0, iOct_global);
    norm = sqrt(df_dx * df_dx + df_dy * df_dy + df_dz * df_dz);

    // if norm is very small, it means the sif is almost uniform
    // avoid division by zero
    if (norm < SMALL_NORM_F)
      norm = ONE_F;

    m_userdata_out(ijk, IX, iOct_global) = df_dx / norm;
    m_userdata_out(ijk, IY, iOct_global) = df_dy / norm;
    m_userdata_out(ijk, IZ, iOct_global) = df_dz / norm;
  }
} // FirstOrderDerivativeFiniteDifference<dim, device_t>::operator() - TagNormalizedGradient

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FirstOrderDerivativeFiniteDifference<dim, device_t>::operator()(TagDivergence const &,
                                                                const index_t & global_index) const
{
  const auto iOct_local = global_index / m_userdata_in.num_cells_inner();
  const auto cell_index = global_index - iOct_local * m_userdata_in.num_cells_inner();

  const auto iOct_global = m_iOct_begin + iOct_local;

  const auto ijk = cellindex_to_coord<dim>(cell_index, m_userdata_in.block_size());

  const auto dfx_dx = compute_first_derivative<IX>(ijk, IX, iOct_global);
  const auto dfy_dy = compute_first_derivative<IY>(ijk, IY, iOct_global);

  auto div = dfx_dx + dfy_dy;

  if constexpr (dim == 3)
  {
    const auto dfz_dz = compute_first_derivative<IZ>(ijk, IZ, iOct_global);
    div += dfz_dz;
  }

  m_userdata_out(ijk, 0, iOct_global) = m_scalar_factor * div;

} // FirstOrderDerivativeFiniteDifference<dim, device_t>::operator() - TagDivergence

// ================================================================================================
// ================================================================================================
template class FirstOrderDerivativeFiniteDifference<2, kalypsso::DefaultDevice>;
template class FirstOrderDerivativeFiniteDifference<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
