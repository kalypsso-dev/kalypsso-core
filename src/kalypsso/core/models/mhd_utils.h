// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file mhd_utils.h
 * \brief Small MHD related utilities common to CPU / GPU code.
 *
 * These utility functions (find_speed_fast, etc...) are directly
 * adapted from Fortran original code found in RAMSES/DUMSES.
 *
 */
#ifndef KALYPSSO_CORE_MODELS_MHD_UTILS_H_
#define KALYPSSO_CORE_MODELS_MHD_UTILS_H_

#include <kalypsso/core/real_type.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/models/MHD.h>

namespace kalypsso
{

namespace core
{

namespace models
{

namespace mhd
{
using MagneticField_t = Kokkos::Array<real_t, 3>;
/**
 * max value out of 4
 */
KOKKOS_INLINE_FUNCTION
real_t
FMAX4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = (a1 > returnVal) ? a1 : returnVal;
  returnVal = (a2 > returnVal) ? a2 : returnVal;
  returnVal = (a3 > returnVal) ? a3 : returnVal;

  return returnVal;
} // FMAX4

/**
 * min value out of 4
 */
KOKKOS_INLINE_FUNCTION
real_t
FMIN4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = (a1 < returnVal) ? a1 : returnVal;
  returnVal = (a2 < returnVal) ? a2 : returnVal;
  returnVal = (a3 < returnVal) ? a3 : returnVal;

  return returnVal;
} // FMIN4

/**
 * max value out of 5
 */
KOKKOS_INLINE_FUNCTION
real_t
FMAX5(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4)
{
  real_t returnVal = a0;
  returnVal = (a1 > returnVal) ? a1 : returnVal;
  returnVal = (a2 > returnVal) ? a2 : returnVal;
  returnVal = (a3 > returnVal) ? a3 : returnVal;
  returnVal = (a4 > returnVal) ? a4 : returnVal;

  return returnVal;
} // FMAX5

/**
 * Compute the fast magnetosonic velocity.
 *
 * IU is index to Vnormal
 * IA is index to Bnormal
 *
 * IV, IW are indexes to Vtransverse1, Vtransverse2,
 * IB, IC are indexes to Btransverse1, Btransverse2
 *
 * \param[in] qvar array of (cell-centered) primitive variables
 * \param[in] mhd_settings struct used to retrieve gamma0 (ideal gas law)
 *
 * \return fast magnetosonic wave speed along given direction.
 */
template <ComponentIndex3D dir>
KOKKOS_INLINE_FUNCTION real_t
find_speed_fast(MHDStateCell const & qvar, MHDSettings const & mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;
  real_t         cf = ZERO_F;

  auto const & d = qvar[MHD::ID];
  auto const & p = qvar[MHD::IP];
  auto const & a = qvar[MHD::IA];
  auto const & b = qvar[MHD::IB];
  auto const & c = qvar[MHD::IC];

  const auto b2 = a * a + b * b + c * c;
  const auto c2 = gamma0 * p / d;
  const auto d2 = HALF_F * (b2 / d + c2);

  if constexpr (dir == IX)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  else if constexpr (dir == IY)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * b * b / d));

  else if constexpr (dir == IZ)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * c * c / d));

  return cf;

} // find_speed_fast

template <ComponentIndex3D dir>
KOKKOS_INLINE_FUNCTION real_t
find_speed_fast(MHDSplitStateCell const & qvar,
                MagneticField_t const &   MF,
                MHDSettings const &       mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;
  real_t         cf = ZERO_F;

  auto const & d = qvar[MHD::ID];
  auto const & p = qvar[MHD::IP];
  auto const & a = qvar[MHD::IA];
  auto const & b = qvar[MHD::IB];
  auto const & c = qvar[MHD::IC];

  auto const & a0 = MF[IX];
  auto const & b0 = MF[IY];
  auto const & c0 = MF[IZ];

  const auto b2 = (a + a0) * (a + a0) + (b + b0) * (b + b0) + (c + c0) * (c + c0);
  const auto c2 = gamma0 * p / d;
  const auto d2 = HALF_F * (b2 / d + c2);


  if constexpr (dir == IX)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * (a + a0) * (a + a0) / d));

  else if constexpr (dir == IY)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * (b + b0) * (b + b0) / d));

  else if constexpr (dir == IZ)
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * (c + c0) * (c + c0) / d));

  return cf;

} // find_speed_fast


/**
 * Compute the Alfven velocity.
 *
 * The structure of qvar is :
 * rho, pressure,
 * vnormal, vtransverse1, vtransverse2,
 * bnormal, btransverse1, btransverse2
 *
 * \param[in] qvar array of (cell-centered) primitive variables
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_alfven(MHDStateCell const & qvar)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  real_t d = qvar[MHD::ID];
  real_t a = qvar[MHD::IA];

  return sqrt(a * a / d);

} // find_speed_alfven

/**
 * Compute the Alfven velocity.
 *
 * Simpler interface.
 * \param[in] d density
 * \param[in] a normal magnetic field\
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_alfven(real_t d, real_t a)
{

  return sqrt(a * a / d);

} // find_speed_alfven

/**
 * Compute the 1d mhd fluxes from the conservative.
 *
 * Only used in Riemann solver HLL (probably cartesian only
 * compatible, since gas pressure is included).
 *
 * variables. The structure of qvar is :
 * rho, pressure,
 * vnormal, vtransverse1, vtransverse2,
 * bnormal, btransverse1, btransverse2.
 *
 * @param[in]  qvar state vector (primitive variables)
 * @param[out] cvar state vector (conservative variables)
 * @param[out] ff flux vector
 *
 */
KOKKOS_INLINE_FUNCTION
void
find_mhd_flux(MHDStateCell const & qvar,
              MHDStateCell &       cvar,
              MHDStateCell &       ff,
              MHDSettings const &  mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  real_t const & gamma0 = mhd_settings.hydro.gamma0;

  // ISOTHERMAL
  const real_t & cIso = mhd_settings.hydro.cIso;
  real_t         p;
  if (cIso > 0)
  {
    // recompute pressure
    p = qvar[MHD::ID] * cIso * cIso;
  }
  else
  {
    p = qvar[MHD::IP];
  }
  // end ISOTHERMAL

  // local variables
  auto const entho = ONE_F / (gamma0 - ONE_F);

  auto const & d = qvar[MHD::ID];
  auto const & u = qvar[MHD::IU];
  auto const & v = qvar[MHD::IV];
  auto const & w = qvar[MHD::IW];
  auto const & a = qvar[MHD::IA];
  auto const & b = qvar[MHD::IB];
  auto const & c = qvar[MHD::IC];

  auto const ecin = HALF_F * (u * u + v * v + w * w) * d;
  auto const emag = HALF_F * (a * a + b * b + c * c);
  auto const etot = p * entho + ecin + emag;
  auto const ptot = p + emag;

  // compute conservative variables
  cvar[MHD::ID] = d;
  cvar[MHD::IP] = etot;
  cvar[MHD::IU] = d * u;
  cvar[MHD::IV] = d * v;
  cvar[MHD::IW] = d * w;
  cvar[MHD::IA] = a;
  cvar[MHD::IB] = b;
  cvar[MHD::IC] = c;

  // compute fluxes
  ff[MHD::ID] = d * u;
  ff[MHD::IP] = (etot + ptot) * u - a * (a * u + b * v + c * w);
  ff[MHD::IU] = d * u * u - a * a + ptot; /* *** WARNING pressure included *** */
  ff[MHD::IV] = d * u * v - a * b;
  ff[MHD::IW] = d * u * w - a * c;
  ff[MHD::IA] = ZERO_F;
  ff[MHD::IB] = b * u - a * v;
  ff[MHD::IC] = c * u - a * w;

} // find_mhd_flux

/**
 * Computes fast magnetosonic wave for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastMagSpeed array containing fast magnetosonic speed along
 * x, y, and z direction.
 *
 * \tparam dim if dim==2, only computes magnetosonic speed along x
 * and y.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
fast_mhd_speed(const MHDStateCell & qState, MHDSettings const & mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  Kokkos::Array<real_t, dim> fastMagSpeed;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;

  const real_t & rho = qState[MHD::ID];
  const real_t & p = qState[MHD::IP];
  /*const real_t& vx  = qState[MHD::IU];
    const real_t& vy  = qState[MHD::IV];
    const real_t& vz  = qState[MHD::IW];*/
  const real_t & bx = qState[MHD::IA];
  const real_t & by = qState[MHD::IB];
  const real_t & bz = qState[MHD::IC];

  real_t mag_perp, alfv, vit_son, som_vit, som_vit2, delta, fast_speed;

  // compute fast magnetosonic speed along X
  mag_perp = (by * by + bz * bz) / rho; // bt ^2 / rho
  alfv = bx * bx / rho;                 // bx / sqrt(4pi*rho)
  vit_son = gamma0 * p / rho;           // sonic contribution :  gamma*P / rho

  som_vit = mag_perp + alfv + vit_son; // whatever direction,
  // always the same
  som_vit2 = som_vit * som_vit;

  delta = fmax(ZERO_F, som_vit2 - 4 * vit_son * alfv);

  fast_speed = HALF_F * (som_vit + sqrt(delta));
  fast_speed = sqrt(fast_speed);

  fastMagSpeed[IX] = fast_speed;

  // compute fast magnetosonic speed along Y
  mag_perp = (bx * bx + bz * bz) / rho;
  alfv = by * by / rho;

  delta = fmax(ZERO_F, som_vit2 - 4 * vit_son * alfv);

  fast_speed = HALF_F * (som_vit + sqrt(delta));
  fast_speed = sqrt(fast_speed);

  fastMagSpeed[IY] = fast_speed;

  // compute fast magnetosonic speed along Z
  if constexpr (dim == 3)
  {
    mag_perp = (bx * bx + by * by) / rho;
    alfv = bz * bz / rho;

    delta = fmax(ZERO_F, som_vit2 - 4 * vit_son * alfv);

    fast_speed = HALF_F * (som_vit + sqrt(delta));
    fast_speed = sqrt(fast_speed);

    fastMagSpeed[IZ] = fast_speed;
  }

  return fastMagSpeed;

} // fast_mhd_speed

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed array containing fastest information speed along
 * x, y, and z direction.
 *
 * Directional information speed being defined as :
 * directional fast magneto speed + fabs(velocity component)
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 * \tparam dim if dim==2, only computes information speed along x
 * and y.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
find_speed_info(MHDStateCell const & qState, MHDSettings const & mhd_settings)
{
  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  Kokkos::Array<real_t, dim> fastInfoSpeed;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;

  const auto & d = qState[MHD::ID];
  const auto & p = qState[MHD::IP];
  const auto & u = qState[MHD::IU];
  const auto & v = qState[MHD::IV];
  const auto & w = qState[MHD::IW];
  const auto & a = qState[MHD::IA];
  const auto & b = qState[MHD::IB];
  const auto & c = qState[MHD::IC];

  // square norm of magnetic field
  const auto b2 = a * a + b * b + c * c;

  // square speed of sound
  const auto c2 = gamma0 * p / d;

  const auto d2 = HALF_F * (b2 / d + c2);

  /*
   * compute fastest info speed along X
   */
  real_t cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  fastInfoSpeed[IX] = cf + fabs(u);

  /*
   * compute fastest info speed along Y
   */
  cf = sqrt(d2 + sqrt(d2 * d2 - c2 * b * b / d));

  fastInfoSpeed[IY] = cf + fabs(v);

  /*
   * compute fastest info speed along Z
   */
  if constexpr (dim == 3)
  {
    cf = sqrt(d2 + sqrt(d2 * d2 - c2 * c * c / d));

    fastInfoSpeed[IZ] = cf + fabs(w);
  } // end THREE_D

  return fastInfoSpeed;

} // find_speed_info

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed fastest information speed along x
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
find_speed_info(MHDStateCell const & qState, MHDSettings const & mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;
  const real_t & u = qState[MHD::IU];
  // const real_t& v = qState[MHD::IV];
  // const real_t& w = qState[MHD::IW];

  auto const & d = qState[MHD::ID];
  auto const & p = qState[MHD::IP];
  auto const & a = qState[MHD::IA];
  auto const & b = qState[MHD::IB];
  auto const & c = qState[MHD::IC];

  // compute fastest info speed along X
  const auto b2 = a * a + b * b + c * c;
  const auto c2 = gamma0 * p / d;
  const auto d2 = HALF_F * (b2 / d + c2);
  const auto cf = sqrt(d2 + sqrt(d2 * d2 - c2 * a * a / d));

  // return value
  return cf + fabs(u);

} // find_speed_info

// ================================================================================
// ================================================================================
/**
 * Compute Boris factor (square)
 * Return the square of the Boris factor:
 * \gamma_A^2 = 1 / (1 + va^2 / c^2)
 *            = 1 / (1 + B^2 / (\mu_0 \rho c^2))
 * \param[in] u conservative variables array (cell-centered)
 * \param[in] mhd_settings struct used to retrieve cboris
 *
 * \return ga2_Boris real_t
 */
KOKKOS_INLINE_FUNCTION auto
compute_Boris(MHDStateCell const & u, MHDSettings const & mhd_settings)
{
  using MHD = kalypsso::core::models::MHD;

  const auto & cboris2 = mhd_settings.cboris * mhd_settings.cboris;

  auto const & d = u[MHD::ID];
  auto const & A = u[MHD::IA];
  auto const & B = u[MHD::IB];
  auto const & C = u[MHD::IC];

  const auto Va2 = ((A * A) + (B * B) + (C * C)) / d;

  // return ga2_Boris
  return ONE_F / (ONE_F + Va2 / cboris2);

} // compute_Boris

// ==========================================================================================
// ==========================================================================================
/**
 * Compute the fast magnetosonic velocity.
 *
 * IU is index to Vnormal
 * IA is index to Bnormal
 *
 * IV, IW are indexes to Vtransverse1, Vtransverse2,
 * IB, IC are indexes to Btransverse1, Btransverse2
 *
 * \param[in] qvar array of (cell-centered) primitive variables
 * \param[in] mhd_settings struct used to retrieve gamma0 (ideal gas law)
 *
 * \return fast magnetosonic wave speed along given direction.
 */
template <ComponentIndex3D dir>
KOKKOS_INLINE_FUNCTION real_t
find_speed_fast_boris(MHDStateCell const & qvar, MHDSettings const & mhd_settings)
{

  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;
  real_t         cf = ZERO_F;

  real_t ga2 = compute_Boris(qvar, mhd_settings);

  auto const & d = qvar[MHD::ID];
  auto const & p = qvar[MHD::IP];
  auto const & a = qvar[MHD::IA];
  auto const & b = qvar[MHD::IB];
  auto const & c = qvar[MHD::IC];

  const auto b2 = a * a + b * b + c * c;
  const auto c2 = gamma0 * p / d;
  const auto d2 = HALF_F * (b2 / d + c2);

  if constexpr (dir == IX)
    cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * a * a / d));

  else if constexpr (dir == IY)
    cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * b * b / d));

  else if constexpr (dir == IZ)
    cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * c * c / d));

  return cf;

} // find_speed_fast_boris

// ======================================================================================
// ======================================================================================
/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed array containing fastest information speed along
 * x, y, and z direction.
 *
 * Directional information speed being defined as :
 * directional fast magneto speed + fabs(velocity component)
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 * \tparam dim if dim==2, only computes information speed along x
 * and y.
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
find_speed_info_boris(MHDStateCell const & qState, MHDSettings const & mhd_settings)
{
  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  Kokkos::Array<real_t, dim> fastInfoSpeed;

  const real_t & gamma0 = mhd_settings.hydro.gamma0;
  real_t         ga2 = compute_Boris(qState, mhd_settings);

  const auto & d = qState[MHD::ID];
  const auto & p = qState[MHD::IP];
  const auto & u = qState[MHD::IU];
  const auto & v = qState[MHD::IV];
  const auto & w = qState[MHD::IW];
  const auto & a = qState[MHD::IA];
  const auto & b = qState[MHD::IB];
  const auto & c = qState[MHD::IC];

  // square norm of magnetic field
  const auto b2 = a * a + b * b + c * c;

  // square speed of sound
  const auto c2 = gamma0 * p / d;

  const auto d2 = HALF_F * (b2 / d + c2);

  /*
   * compute fastest info speed along X
   */
  real_t cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * a * a / d));

  fastInfoSpeed[IX] = cf + fabs(u);

  /*
   * compute fastest info speed along Y
   */
  cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * b * b / d));

  fastInfoSpeed[IY] = cf + fabs(v);

  /*
   * compute fastest info speed along Z
   */
  if constexpr (dim == 3)
  {
    cf = sqrt(ga2 * d2 + sqrt(d2 * d2 - c2 * c * c / d));

    fastInfoSpeed[IZ] = cf + fabs(w);
  } // end THREE_D

  return fastInfoSpeed;

} // find_speed_info_boris

// ================================================================================
// ================================================================================
/**
 *
 * Convert conservative variables with boris correction:
 * - (rho, rho*u / ga2, rho*v/ ga2, rho*w /ga2, e, bxc, byc, bzc)
 * to primitive variables with boris correction:
 * - (rho, u/ga2, v/ga2, w/ga2, p, bxc, byc, bzc)
 *
 * \param[in]  u  conservative variables array (cell-centered)
 * \param[out] c  speed of sound
 * \param[in]  mhd_settings struct used to retrieve gamma0 (ideal gas law)
 *
 * \return     q  primitive    variables array (cell-centered)
 */
KOKKOS_INLINE_FUNCTION auto
computePrimitives_Boris(MHDStateCell const & u, real_t & c, MHDSettings const & mhd_settings)
{
  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  const auto & gamma0 = mhd_settings.hydro.gamma0;
  const auto & smallr = mhd_settings.hydro.smallr;
  const auto & smallp = mhd_settings.hydro.smallp;

  MHDStateCell q;

  real_t ga2 = compute_Boris(u, mhd_settings);

  // density
  q[MHD::ID] = fmax(u[MHD::ID], smallr);

  // velocity
  q[MHD::IU] = u[MHD::IU] / q[MHD::ID];
  q[MHD::IV] = u[MHD::IV] / q[MHD::ID];
  q[MHD::IW] = u[MHD::IW] / q[MHD::ID];

  // compute cell-centered magnetic field
  q[MHD::IA] = u[MHD::IA];
  q[MHD::IB] = u[MHD::IB];
  q[MHD::IC] = u[MHD::IC];

  //
  // kinetic energy, must be calculated without the boris correction
  //
  // clang-format off
  const auto eken = HALF_F * ((u[MHD::IU] * ga2 / q[MHD::ID] ) * (u[MHD::IU] * ga2 / q[MHD::ID] ) +
                              (u[MHD::IV] * ga2 / q[MHD::ID] ) * (u[MHD::IV] * ga2 / q[MHD::ID] ) +
                              (u[MHD::IW] * ga2 / q[MHD::ID] ) * (u[MHD::IW] * ga2 / q[MHD::ID] ));
  // clang-format on


  //
  // magnetic energy
  //
  // clang-format off
  const auto emag = HALF_F * (q[MHD::IA] * q[MHD::IA] +
                              q[MHD::IB] * q[MHD::IB] +
                              q[MHD::IC] * q[MHD::IC]);
  // clang-format on

  // internal energy
  const auto eint = (u[MHD::IE] - emag) / q[MHD::ID] - eken;

  // compute pressure and speed of sound
  if (mhd_settings.hydro.cIso > 0)
  {
    // isothermal
    q[MHD::IP] = q[MHD::ID] * (mhd_settings.hydro.cIso) * (mhd_settings.hydro.cIso);
    c = mhd_settings.hydro.cIso;
  }
  else
  {
    // perfect gas
    q[MHD::IP] = fmax((gamma0 - ONE_F) * q[MHD::ID] * eint, q[MHD::ID] * smallp);
    c = sqrt(mhd_settings.hydro.gamma0 * q[MHD::IP] / q[MHD::ID]);
  }

  return q;

} // computePrimitives_Boris

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables:
 * - (rho, rho*u, rho*v, rho*w, e, , bxc, byc, bzc)
 * to primitive variables:
 * - (rho, u, v, w, p, bxc, byc, bzc)
 *
 * \param[in]  u  conservative variables array (cell-centered)
 * \param[out] c  speed of sound
 * \param[in]  mhd_settings struct used to retrieve gamma0 (ideal gas law)
 *
 * \return     q  primitive    variables array (cell-centered)
 */
KOKKOS_INLINE_FUNCTION auto
computePrimitives(MHDStateCell const & u, real_t & c, MHDSettings const & mhd_settings)
{
  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const auto & gamma0 = mhd_settings.hydro.gamma0;
  const auto & smallr = mhd_settings.hydro.smallr;
  const auto & smallp = mhd_settings.hydro.smallp;

  MHDStateCell q;

  // density
  q[MHD::ID] = fmax(u[MHD::ID], smallr);

  // velocity
  q[MHD::IU] = u[MHD::IU] / q[MHD::ID];
  q[MHD::IV] = u[MHD::IV] / q[MHD::ID];
  q[MHD::IW] = u[MHD::IW] / q[MHD::ID];

  // compute cell-centered magnetic field
  q[MHD::IA] = u[MHD::IA];
  q[MHD::IB] = u[MHD::IB];
  q[MHD::IC] = u[MHD::IC];

  //
  // kinetic energy
  //
  // clang-format off
  const auto eken = HALF_F * (q[MHD::IU] * q[MHD::IU] +
                              q[MHD::IV] * q[MHD::IV] +
                              q[MHD::IW] * q[MHD::IW]);
  // clang-format on

  //
  // magnetic energy
  //
  // clang-format off
  const auto emag = HALF_F * (q[MHD::IA] * q[MHD::IA] +
                              q[MHD::IB] * q[MHD::IB] +
                              q[MHD::IC] * q[MHD::IC]);
  // clang-format on

  // internal energy
  const auto eint = (u[MHD::IE] - emag) / q[MHD::ID] - eken;
  if (mhd_settings.hydro.abort_when_negative_eint and eint < 0)
  {
    Kokkos::abort("Negative internal energy detected - can't proceed further");
  }

  // compute pressure and speed of sound
  if (mhd_settings.hydro.cIso > 0)
  {
    // isothermal
    q[MHD::IP] = q[MHD::ID] * (mhd_settings.hydro.cIso) * (mhd_settings.hydro.cIso);
    c = mhd_settings.hydro.cIso;
  }
  else
  {
    // perfect gas
    q[MHD::IP] = fmax((gamma0 - ONE_F) * q[MHD::ID] * eint, q[MHD::ID] * smallp);
    c = sqrt(mhd_settings.hydro.gamma0 * q[MHD::IP] / q[MHD::ID]);
  }

  return q;

} // computePrimitives

// ================================================================================
// ================================================================================

KOKKOS_INLINE_FUNCTION auto
computePrimitives(MHDSplitStateCell const &   u,
                  real_t &                    c,
                  MHDSettings const &         mhd_settings,
                  FieldMap<core::models::MHD> fm)
{
  // makes enum MHD::VarId available
  using MHD = kalypsso::core::models::MHD;

  const auto & gamma0 = mhd_settings.hydro.gamma0;
  const auto & smallr = mhd_settings.hydro.smallr;
  const auto & smallp = mhd_settings.hydro.smallp;

  MHDSplitStateCell q;

  // density
  q[fm[MHD::ID]] = fmax(u[fm[MHD::ID]], smallr);

  // velocity
  q[fm[MHD::IU]] = u[fm[MHD::IU]] / q[fm[MHD::ID]];
  q[fm[MHD::IV]] = u[fm[MHD::IV]] / q[fm[MHD::ID]];
  q[fm[MHD::IW]] = u[fm[MHD::IW]] / q[fm[MHD::ID]];

  // compute cell-centered magnetic field
  q[fm[MHD::IA]] = u[fm[MHD::IA]];
  q[fm[MHD::IB]] = u[fm[MHD::IB]];
  q[fm[MHD::IC]] = u[fm[MHD::IC]];

  // compute cell-centered zer-order magnetic field
  q[fm[MHD::IA0]] = u[fm[MHD::IA0]];
  q[fm[MHD::IB0]] = u[fm[MHD::IB0]];
  q[fm[MHD::IC0]] = u[fm[MHD::IC0]];

  //
  // kinetic energy
  //
  // clang-format off
  const auto eken = HALF_F * (q[fm[MHD::IU]] * q[fm[MHD::IU]] +
                              q[fm[MHD::IV]] * q[fm[MHD::IV]] +
                              q[fm[MHD::IW]] * q[fm[MHD::IW]]);
  // clang-format on

  //
  // magnetic energy
  //
  // clang-format off
  const auto emag = HALF_F * (q[fm[MHD::IA]] * q[fm[MHD::IA]] +
                              q[fm[MHD::IB]] * q[fm[MHD::IB]] +
                              q[fm[MHD::IC]] * q[fm[MHD::IC]]);

  // clang-format on

  // internal energy
  const auto eint = (u[MHD::IE] - emag) / q[MHD::ID] - eken;

  // compute pressure and speed of sound
  if (mhd_settings.hydro.cIso > 0)
  {
    // isothermal
    q[MHD::IP] = q[MHD::ID] * (mhd_settings.hydro.cIso) * (mhd_settings.hydro.cIso);
    c = mhd_settings.hydro.cIso;
  }
  else
  {
    // perfect gas
    q[MHD::IP] = fmax((gamma0 - ONE_F) * q[MHD::ID] * eint, q[MHD::ID] * smallp);
    c = sqrt(mhd_settings.hydro.gamma0 * q[MHD::IP] / q[MHD::ID]);
  }

  return q;

} // computePrimitives

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables:
 * - (rho, rho*u, rho*v, rho*w, e, , bxc, byc, bzc)
 * to primitive variables:
 * - (rho, u, v, w, p, bxc, byc, bzc)
 *
 * \param[in]  u  conservative variables array (cell-centered)
 * \param[out] c  speed of sound
 * \param[in]  mhd_settings struct used to retrieve gamma0 (ideal gas law)
 *
 * \return     q  primitive    variables array (cell-centered)
 *
 * \note here we don't care about returning the speed of sound
 */
KOKKOS_INLINE_FUNCTION auto
computePrimitives_Boris(MHDStateCell const & u, MHDSettings const & mhd_settings)
{
  [[maybe_unused]] real_t c;
  return computePrimitives_Boris(u, c, mhd_settings);
} // computePrimitives

KOKKOS_INLINE_FUNCTION auto
computePrimitives(MHDStateCell const & u, MHDSettings const & mhd_settings)
{
  [[maybe_unused]] real_t c;
  return computePrimitives(u, c, mhd_settings);
} // computePrimitives

KOKKOS_INLINE_FUNCTION auto
computePrimitives(MHDSplitStateCell const &   u,
                  MHDSettings const &         mhd_settings,
                  FieldMap<core::models::MHD> fm)
{
  [[maybe_unused]] real_t c;
  return computePrimitives(u, c, mhd_settings, fm);
} // computePrimitives

} // namespace mhd

} // namespace models

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_MHD_UTILS_H_
