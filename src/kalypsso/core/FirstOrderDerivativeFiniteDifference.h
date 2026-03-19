// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FirstOrderDerivativeFiniteDifference.h
 *
 * First order derivative implementation.
 */
#ifndef KALYPSSO_CORE_FIRSTORDERDERIVATIVEFINITEDIFFERENCE_H_
#define KALYPSSO_CORE_FIRSTORDERDERIVATIVEFINITEDIFFERENCE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/FiniteDifferenceData.h>

#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

namespace core
{

// =============================================================================
// =============================================================================
/**
 * \class FirstOrderDerivativeFiniteDifference
 *
 * Implement first order derivative using a N points finite difference approximation using only
 * points inside the same AMR block of cells.
 * We support either N=2 or N=4 points.
 *
 * When N=2 points, the canonical stencil is to use two points equispaced from the central points
 * interpreted as cell-centered values (see drawing below):
 *
 * Central stencil
 *  ___________________________________
 * |          |           |           |
 * |          |           |           |
 * |   x[0]   |    x_c    |    x[1]   |
 * |          |           |           |
 * |__________|___________|___________|
 *
 * When left and right neighbor are at the same AMR level), the usual second order approximation
 * reads : \f$ \frac{df}{dx} = \frac{f(x+h) - f(x-h)}{2h} + \mathcal{O}(h^2)\f$, where
 * \f$ h \f$  is the central cell size.
 *
 * This formula can be recast into \f$ \frac{df}{dx} = \sum_{i=0}^{i=1} c_i * (f(x_i)-f(x_c))/h
 * where \f$ x_c\f$ is the central point, \f$ c_0=-1/2\f$, \f$ c_1=1/2\f$ are the finite
 * difference coefficients and \f$ x_0 = x_c-h \f$ and \f$ x_1=x_c+h \f$.
 *
 * Forward stencil
 *  ___________________________________
 * |          |           |           |
 * |          |           |           |
 * |   x_c    |    x[0]   |    x[1]   |
 * |          |           |           |
 * |__________|___________|___________|
 *
 * When the point where we want to compute derivative is touching the block border, one switch to a
 * forward finite difference formula.
 * The new coefficients can be computed using the utility file stencil_ceofs_helper.py
 *
 * Using length in units of h (current cell size), and taking origin at x_c, then x[0]=1.0
 * and x[1]=2.0, the finite difference coefficients can be obtained using our
 * stencil_coefs_helper.py
 *
 * ./stencil_coefs_helper.py -o 1 -p 1.0 2.0
 *
 * from which we obtain: coeff[0]=2.0 and coeff[1]=-1/2
 *
 * Backward stencil
 *  ___________________________________
 * |          |           |           |
 * |          |           |           |
 * |   x[0]   |    x[1]   |    x_c    |
 * |          |           |           |
 * |__________|___________|___________|
 *
 * ./stencil_coefs_helper.py -o 1 -p -2.0 -1.0
 *
 * from which we obtain: coeff[0]=1/2 and coeff[1]=2
 *
 * \tparam dim is dimension (2 or 3)
 * \tparam device_t is a kokkos device
 *
 * \note input and output data are DataArrayGhostedBlock eventhough it would only require
 * DataArrayBlock since we only compute the inner part of the ghosted block. Most downstream user
 * code of this class will actually use ghosted array.
 */
template <size_t dim, typename device_t>
class FirstOrderDerivativeFiniteDifference
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  //! Compute first derivative
  struct TagFirstDerivative
  {};

  //! Compute (normalized) gradient
  struct TagNormalizedGradient
  {};

  //! Compute divergence
  struct TagDivergence
  {};

  // ====================================================================
  // ====================================================================
  //! constructor.
  FirstOrderDerivativeFiniteDifference(ConfigMap const &        config_map,
                                       orchard_key_view_t       orchard_keys,
                                       int32_t                  iOct_begin,
                                       int32_t                  num_octants,
                                       DataArrayGhostedBlock_t  userdata_in,
                                       int32_t                  ivar_in,
                                       DataArrayGhostedBlock_t  userdata_out,
                                       int32_t                  ivar_out,
                                       int32_t                  derivative_dir,
                                       FIRST_DERIVATIVE_STENCIL stencil_length,
                                       real_t                   scalar_factor);

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~FirstOrderDerivativeFiniteDifference() = default;

  // ====================================================================
  // ====================================================================
  static void
  check_args_validity(DataArrayGhostedBlock_t  userdata_in,
                      DataArrayGhostedBlock_t  userdata_out,
                      FIRST_DERIVATIVE_STENCIL stencil_length);

  // ====================================================================
  // ====================================================================
  static void
  first_derivative(ConfigMap const &        config_map,
                   orchard_key_view_t       orchard_keys,
                   int32_t                  iOct_begin,
                   int32_t                  num_octants,
                   DataArrayGhostedBlock_t  userdata_in,
                   int32_t                  ivar_in,
                   DataArrayGhostedBlock_t  userdata_out,
                   int32_t                  ivar_out,
                   int32_t                  derivative_dir,
                   FIRST_DERIVATIVE_STENCIL stencil_length);

  // ====================================================================
  // ====================================================================
  static void
  normalized_gradient(ConfigMap const &        config_map,
                      orchard_key_view_t       orchard_keys,
                      int32_t                  iOct_begin,
                      int32_t                  num_octants,
                      DataArrayGhostedBlock_t  userdata_in,
                      int32_t                  ivar_in,
                      DataArrayGhostedBlock_t  userdata_out,
                      FIRST_DERIVATIVE_STENCIL stencil_length);

  // ====================================================================
  // ====================================================================
  static void
  divergence(ConfigMap const &        config_map,
             orchard_key_view_t       orchard_keys,
             int32_t                  iOct_begin,
             int32_t                  num_octants,
             DataArrayGhostedBlock_t  userdata_in,
             DataArrayGhostedBlock_t  userdata_out,
             FIRST_DERIVATIVE_STENCIL stencil_length,
             real_t                   scalar_factor);

  // ==============================================================
  // ==============================================================
  /**
   * Compute first derivative along direction dir using a 3 point stencil.
   *
   * Derivative is estimated with big O(h^2) approximation.
   */
  template <int32_t dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_first_derivative_3_points(coord_t<dim> const & ijk,
                                    int                  ivar,
                                    int32_t const &      iOct_local) const
  {

    KOKKOS_ASSERT(dir < dim && "wrong direction");

    const auto & b = m_userdata_in.block_size();

    const auto iOct_global = m_iOct_begin + iOct_local;

    const auto key = m_keys(iOct_global);
    const auto dx = compute_cell_length<dim>(key, b[dir]) * m_scaling_factor;

    real_t derivative = 0.0;

    // current cell value
    const auto data_C = m_userdata_in(ijk, ivar, iOct_global);

    // current position inside stencil
    int32_t pos = 1;

    if (ijk[dir] == 0)
    {
      pos = 0;
    }
    else if (ijk[dir] == b[dir] - 1)
    {
      pos = 2;
    }

    // s is a stencil position
    for (int s = 0; s < 3; ++s)
    {
      auto ijk_s = ijk;
      ijk_s[dir] += (s - pos);

      derivative += m_fd.coef3(pos, s) * (m_userdata_in(ijk_s, ivar, iOct_global) - data_C) / dx;
    }

    return derivative;

  } // compute_first_derivative_3_points

  // ==============================================================
  // ==============================================================
  /**
   * Compute first derivative along direction dir using a 5 point stencil.
   *
   * Derivative is estimated with big O(h^4) approximation.
   */
  template <int32_t dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_first_derivative_5_points(coord_t<dim> const & ijk,
                                    int                  ivar,
                                    int32_t const &      iOct_local) const
  {

    KOKKOS_ASSERT(dir < dim && "wrong direction");

    const auto & b = m_userdata_in.block_size();

    const auto iOct_global = m_iOct_begin + iOct_local;

    const auto key = m_keys(iOct_global);
    const auto dx = compute_cell_length<dim>(key, b[dir]) * m_scaling_factor;

    real_t derivative = 0.0;

    // current cell value
    const auto data_C = m_userdata_in(ijk, ivar, iOct_global);

    // current position inside stencil
    int32_t pos = 2;

    if (ijk[dir] == 0)
    {
      pos = 0;
    }
    else if (ijk[dir] == 1)
    {
      pos = 1;
    }
    else if (ijk[dir] == b[dir] - 2)
    {
      pos = 3;
    }
    else if (ijk[dir] == b[dir] - 1)
    {
      pos = 4;
    }

    // s is a stencil position
    for (int s = 0; s < 5; ++s)
    {
      auto ijk_s = ijk;
      ijk_s[dir] += (s - pos);

      derivative += m_fd.coef5(pos, s) * (m_userdata_in(ijk_s, ivar, iOct_global) - data_C) / dx;
    }

    return derivative;

  } // compute_first_derivative_5_points

  // ==============================================================
  // ==============================================================
  /**
   * Compute first derivative along direction dir using a 7 point stencil.
   *
   * Derivative is estimated with big O(h^6) approximation.
   */
  template <int32_t dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_first_derivative_7_points(coord_t<dim> const & ijk,
                                    int                  ivar,
                                    int32_t const &      iOct_local) const
  {

    KOKKOS_ASSERT(dir < dim && "wrong direction");

    const auto & b = m_userdata_in.block_size();

    const auto iOct_global = m_iOct_begin + iOct_local;

    const auto key = m_keys(iOct_global);
    const auto dx = compute_cell_length<dim>(key, b[dir]) * m_scaling_factor;

    real_t derivative = 0.0;

    // current cell value
    const auto data_C = m_userdata_in(ijk, ivar, iOct_global);

    // current position inside stencil
    int32_t pos = 3;

    if (ijk[dir] == 0)
    {
      pos = 0;
    }
    else if (ijk[dir] == 1)
    {
      pos = 1;
    }
    else if (ijk[dir] == 2)
    {
      pos = 2;
    }
    else if (ijk[dir] == b[dir] - 3)
    {
      pos = 4;
    }
    else if (ijk[dir] == b[dir] - 2)
    {
      pos = 5;
    }
    else if (ijk[dir] == b[dir] - 1)
    {
      pos = 6;
    }

    // s is a stencil position
    for (int s = 0; s < 7; ++s)
    {
      auto ijk_s = ijk;
      ijk_s[dir] += (s - pos);

      derivative += m_fd.coef7(pos, s) * (m_userdata_in(ijk_s, ivar, iOct_global) - data_C) / dx;
    }

    return derivative;

  } // compute_first_derivative_7_points

  // ==============================================================
  // ==============================================================
  template <int32_t dir>
  KOKKOS_INLINE_FUNCTION real_t
  compute_first_derivative(coord_t<dim> const & ijk, int ivar, int32_t const & iOct_local) const
  {

    if (m_stencil_length == +FIRST_DERIVATIVE_STENCIL::THREE_POINTS)
    {
      return compute_first_derivative_3_points<dir>(ijk, ivar, iOct_local);
    }
    else if (m_stencil_length == +FIRST_DERIVATIVE_STENCIL::FIVE_POINTS)
    {
      return compute_first_derivative_5_points<dir>(ijk, ivar, iOct_local);
    }
    else if (m_stencil_length == +FIRST_DERIVATIVE_STENCIL::SEVEN_POINTS)
    {
      return compute_first_derivative_7_points<dir>(ijk, ivar, iOct_local);
    }

    // we shouldn't be here
    return 0.0;
  }

  // ==============================================================
  // ==============================================================
  /**
   * Range policy functor for computing just the first derivative of a scalar field along a given
   * direction.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFirstDerivative const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * Range policy functor for computing normalized gradient.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagNormalizedGradient const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * Range policy functor for computing divergence of a vector field.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagDivergence const &, const index_t & global_index) const;

private:
  //! Array of orchard keys.
  orchard_key_view_t m_keys;

  //! Starting octant id
  const int32_t m_iOct_begin;

  //! Number of octant to process, starting at m_iOct_begin.
  const int32_t m_num_octants;

  //! Input block data array
  DataArrayGhostedBlock_t m_userdata_in;

  //! index to variable in input array used to compute derivative
  int32_t m_ivar_in;

  //! a block data array (no ghosts)
  DataArrayGhostedBlock_t m_userdata_out;

  //! index to variable in output array where to write the result (derivative)
  //! only when tag is TagFirstDerivative
  int32_t m_ivar_out;

  //! Direction along which derivative is computed
  int32_t m_derivative_dir;

  //! stencil length
  const FIRST_DERIVATIVE_STENCIL m_stencil_length;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! Finite difference coefficients
  const FiniteDifferenceData m_fd;

  //! Scalar factor used in computing divergence
  const real_t m_scalar_factor;

  //! small norm
  KALYPSSO_STATIC_MATH_CONSTANT(SMALL_NORM, 1e-13);

}; // class FirstOrderDerivativeFiniteDifference

extern template class FirstOrderDerivativeFiniteDifference<2, kalypsso::DefaultDevice>;
extern template class FirstOrderDerivativeFiniteDifference<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_FIRSTORDERDERIVATIVEFINITEDIFFERENCE_H_
