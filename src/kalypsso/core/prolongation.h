// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file prolongation.h
 *
 * Define parameters used in prologation operators (from coarse to fine AMR level interpolation).
 */
#ifndef KALYPSSO_CORE_PROLONGATION_H_
#define KALYPSSO_CORE_PROLONGATION_H_

#include <../better-enums/enum.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/log/kalypsso_log.h>

namespace kalypsso
{

// clang-format off
/**
 * Enumerate types of prolongation operator for cell centered values.
 *
 * - SIMPLE_COPY: just copy the coarse cell value into the fine cell
 * - EXTRAPOLATE_LINEAR_MINMOD: use limiting slopes to extrapolate the coarse value to fine
 * - CONSERVATIVE_INTERPOLATION_ORDER_2: use a 3x3 stencil at coarse level to compute
 *        a conservative interpolate value in the fine cells; interpolation is done in
 *        a directionally split way
 * - CONSERVATIVE_INTERPOLATION_ORDER_4: use a 5x5 stencil at coarse level to compute
 *        a conservative interpolate value in the fine cells; interpolation is done in
 *        a directionally split way
 *
 * \note conservative interpolation coefficients are computed using python script :
 * src/shared/utils/conservative_polynomial_interpolation_order_2.py
 */
BETTER_ENUM(CellCenteredProlongationType, int,
            SIMPLE_COPY,
            EXTRAPOLATE_LINEAR_MINMOD,
            CONSERVATIVE_INTERPOLATION_ORDER_2,
            CONSERVATIVE_INTERPOLATION_ORDER_4)
// clang-format on

inline CellCenteredProlongationType
get_cell_prolongation_type(ConfigMap const & config_map)
{
  auto prolongation_name =
    config_map.getString("amr", "cell_centered_prolongation_type", "SIMPLE_COPY");
  auto maybe_value = CellCenteredProlongationType::_from_string_nothrow(prolongation_name.c_str());
  if (maybe_value)
    return *maybe_value;

  KALYPSSO_ERROR("Invalid cell_centered_prolongation_type : {}; using SIMPLE_COPY instead.",
                 prolongation_name.c_str());
  return CellCenteredProlongationType::SIMPLE_COPY;
}

/**
 * Enumerate types of prolongation operator for face centered values (magnetic field) on internal
 * faces.
 *
 * Doing prolongation means provide an algorithm to fill the new fine AMR level face-centered
 * values. The number of new fine AMR level faces is:
 * - in 2d, 2^(dim-1) * 3 *  dim = 2 * 3 * 2 = 12 new faces
 * - in 3d, 2^(dim-1) * 3 *  dim = 4 * 3 * 3 = 36 new faces
 *
 * We must also separate the new fine face into 2 classes:
 * - external faces: the ones that are collocated with faces of the parent coarse cells
 * - internal faces: the ones that are not collocated with faces of parent cells
 *
 * Number of internal/external faces:
 * - in 2d, 2 * dim = 4 internal faces  + 8 external faces
 * - in 3d, 4 * dim = 12 internal faces + 24 external faces
 *
 * Generally speaking, we will face-centered values of the fine cells in two steps:
 *
 * 1. fill the external faces from values of the corresponding coarse cell face, doing, e.g.:
 * - a simple copy
 * - a linear extrapolation using limited transverse slopes
 *
 * 2. fill the internal faces using a divergence preserving algorithm (see reference
 * below):
 * - use formulas from Toth and Roe (2002)
 * - use divergence-preserving B-spline interpolation from Schroeder et al, 2022
 * - other ideas adapted from CEA AMR internal code.
 *
 * The following drawing illustrates internal versus external faces.
 * - on the left  In 2d, there are 4 internal faces, the ones marked with "+", while all other
 * faces, are faces collocated with faces of the old coarse cells.
 *
 *
 * ________________             _________________
 * |              |             |       +       |
 * |              |             |       +       |
 * |              |             |       +       |
 * |              |    ==>      ++++++++ ++++++++
 * |              |             |       +       |
 * |              |             |       +       |
 * |______________|             |_______+______ |
 *
 * Currently implemented:
 *
 * - formulas based on Toth and Roe, JCP, 180, 746-759, 2002: Divergence- and curl-preserving
 * prolongation and restriction formulas. https://doi.org/10.1006/jcph.2002.7120
 *
 * Possible other implementations: use B-spline interpolation approach described in
 *
 * - "Higher order divergence-free and curl-free interpolation on MAC grids", Ray-Chowdurry et al,
 * JCP, vol 503, 112831, 2024
 * - "Local divergence-free polynomial interpolation on MAC grids", Schroeder et al, JCP, vol 468,
 * 111500, 2022
 */
// clang-format off
BETTER_ENUM(FaceCenteredProlongationInternalType, int,
            TOTH_AND_ROE)
// clang-format on

// clang-format off
BETTER_ENUM(FaceCenteredProlongationExternalType, int,
            SIMPLE_COPY,
            EXTRAPOLATE_LINEAR_MINMOD)
// clang-format on

inline FaceCenteredProlongationInternalType
get_internal_face_prolongation_type(ConfigMap const & config_map)
{
  auto prolongation_name =
    config_map.getString("amr", "face_centered_prolongation_internal_type", "TOTH_AND_ROE");
  auto maybe_value =
    FaceCenteredProlongationInternalType::_from_string_nothrow(prolongation_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return FaceCenteredProlongationInternalType::TOTH_AND_ROE;
}

inline FaceCenteredProlongationExternalType
get_external_face_prolongation_type(ConfigMap const & config_map)
{
  auto prolongation_name =
    config_map.getString("amr", "face_centered_prolongation_external_type", "SIMPLE_COPY");
  auto maybe_value =
    FaceCenteredProlongationExternalType::_from_string_nothrow(prolongation_name.c_str());
  if (maybe_value)
    return *maybe_value;
  return FaceCenteredProlongationExternalType::SIMPLE_COPY;
}

// ==============================================================
// ==============================================================
struct ProlongationParam
{

  CellCenteredProlongationType         m_cell;
  FaceCenteredProlongationInternalType m_face_internal;
  FaceCenteredProlongationExternalType m_face_external;

  ProlongationParam(ConfigMap const & config_map)
    : m_cell(get_cell_prolongation_type(config_map))
    , m_face_internal(get_internal_face_prolongation_type(config_map))
    , m_face_external(get_external_face_prolongation_type(config_map))
  {}

}; // struct ProlongationParam

// ==============================================================
// ==============================================================
KOKKOS_INLINE_FUNCTION real_t
minmod_scalar(real_t q, real_t qPlus, real_t qMinus, real_t slope_type)
{
  real_t dq = 0;

  if (slope_type == 1 or slope_type == 2)
  {
    const real_t dlft = slope_type * (q - qMinus);
    const real_t drgt = slope_type * (qPlus - q);
    const real_t dcen = HALF_F * (qPlus - qMinus);
    const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    const real_t slop = fmin(fabs(dlft), fabs(drgt));
    const real_t dlim = (dlft * drgt) <= ZERO_F ? ZERO_F : slop;
    dq = dsgn * fmin(dlim, fabs(dcen));
  }

  return dq;

} // minmod_scalar

/**
 * A companion structure for performing conservative interpolation.
 */
struct ConservativeInterpolation
{
  const Kokkos::Array<real_t, 3> COEFS2_L{ KALYPSSO_NUM(0.125),
                                           KALYPSSO_NUM(1.0),
                                           KALYPSSO_NUM(-0.125) };

  const Kokkos::Array<real_t, 3> COEFS2_R{ KALYPSSO_NUM(-0.125),
                                           KALYPSSO_NUM(1.0),
                                           KALYPSSO_NUM(0.125) };


  //! Perform second order conservative interpolation.
  //!
  //! Consider threa integral values corresponding to three consecutive cells:
  //!
  //! |   Im1    |     I0    |    Ip1  |
  //! |          |           |         |
  //! |          |  L  |  R  |         |
  //!
  //! we want to interpolate conservative values in the fine cell (central cell refined)
  //!
  //! interpolation coefficients are obtained from
  //! utils/conservative_polynomial_interpolation_order_2.py
  KOKKOS_INLINE_FUNCTION real_t
  order2(real_t const & Im1, real_t const & I0, real_t const & Ip1, bool is_left) const
  {
    return is_left ? COEFS2_L[0] * Im1 + COEFS2_L[1] * I0 + COEFS2_L[2] * Ip1
                   : COEFS2_R[0] * Im1 + COEFS2_R[1] * I0 + COEFS2_R[2] * Ip1;
  }

  const Kokkos::Array<real_t, 5> COEFS4_L{ KALYPSSO_NUM(-3.0) / 128,
                                           KALYPSSO_NUM(11.0) / 64,
                                           KALYPSSO_NUM(1.0),
                                           KALYPSSO_NUM(-11.0) / 64,
                                           KALYPSSO_NUM(3.0) / 128 };

  const Kokkos::Array<real_t, 5> COEFS4_R{ KALYPSSO_NUM(3.0) / 128,
                                           KALYPSSO_NUM(-11.0) / 64,
                                           KALYPSSO_NUM(1.0),
                                           KALYPSSO_NUM(11.0) / 64,
                                           KALYPSSO_NUM(-3.0) / 128 };

  //! Perform fourth order conservative interpolation.
  //!
  //! Consider threa integral values corresponding to three consecutive cells:
  //!
  //! |   Im2    |     Im1   |    I0     |     Ip1   |    Ip2   |
  //! |          |           |           |           |          |
  //! |          |           |  L  |  R  |           |          |
  //!
  //! we want to interpolate conservative values in the fine cell (central cell refined)
  //!
  //! interpolation coefficients are obtained from
  //! utils/conservative_polynomial_interpolation_order_4.py
  KOKKOS_INLINE_FUNCTION real_t
  order4(real_t const & Im2,
         real_t const & Im1,
         real_t const & I0,
         real_t const & Ip1,
         real_t const & Ip2,
         bool           is_left) const
  {
    return is_left ? COEFS4_L[0] * Im2 + COEFS4_L[1] * Im1 + COEFS4_L[2] * I0 + COEFS4_L[3] * Ip1 +
                       COEFS4_L[4] * Ip2
                   : COEFS4_R[0] * Im2 + COEFS4_R[1] * Im1 + COEFS4_R[2] * I0 + COEFS4_R[3] * Ip1 +
                       COEFS4_R[4] * Ip2;
  }

}; // struct ConservativeInterpolation

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROLONGATION_H_
