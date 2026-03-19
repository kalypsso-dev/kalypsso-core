// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RiemannSolvers.h
 *
 * Several Riemann solvers.
 */
#ifndef KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_H_
#define KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_H_

#include <math.h>

#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroSettings.h>
#include <kalypsso/core/models/riemann_solver_types.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/eos/eos_utils.h>

namespace kalypsso
{
/**
 * Compute cell fluxes from the Godunov state
 * \param[in]  qgdnv input primitive variables Godunov state
 * \param[out] flux  output flux vector
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
cmpflx(HydroState<dim> const & qgdnv, HydroState<dim> & flux, EosWrapper const & eos)
{
  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // Compute fluxes
  // Mass density
  flux[Hydro::ID] = qgdnv[Hydro::ID] * qgdnv[Hydro::IU];

  // Normal momentum
  flux[Hydro::IU] = flux[Hydro::ID] * qgdnv[Hydro::IU] + qgdnv[Hydro::IP];

  // Transverse momentum 1
  flux[Hydro::IV] = flux[Hydro::ID] * qgdnv[Hydro::IV];

  if constexpr (dim == 3)
    flux[Hydro::IW] = flux[Hydro::ID] * qgdnv[Hydro::IW];

  // Total energy
  real_t ekin;
  ekin = HALF_F * qgdnv[Hydro::ID] *
         (qgdnv[Hydro::IU] * qgdnv[Hydro::IU] + qgdnv[Hydro::IV] * qgdnv[Hydro::IV]);
  if constexpr (dim == 3)
    ekin += HALF_F * qgdnv[Hydro::ID] * (qgdnv[Hydro::IW] * qgdnv[Hydro::IW]);

  const auto etot = eos.volumic_eint_from_pressure(qgdnv[Hydro::IP], qgdnv[Hydro::ID]) + ekin;
  flux[Hydro::IP] = qgdnv[Hydro::IU] * (etot + qgdnv[Hydro::IP]);

} // cmpflx

/**
 * Riemann solver, equivalent to riemann_approx in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * @param[in] qL     : input left  state (primitive variables)
 * @param[in] qR     : input right state (primitive variables)
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_approx(HydroState<dim> const & qL,
               HydroState<dim> const & qR,
               HydroState<dim> &       flux,
               HydroSettings const &   settings,
               EosWrapper const &      eos)
{
  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  HydroState<dim> qgdnv;

  // this is only valid for ideal gas eos
  const auto gamma6 = core::eos::gamma6(eos.gamma());
  KOKKOS_ASSERT(gamma6 > 0);

  auto const & smallr = settings.smallr;
  auto const & smallc = settings.smallc;
  auto const & smallp = settings.smallp;
  auto const & smallpp = settings.smallpp;

  // Pressure, density and velocity
  const auto rl = fmax(qL[Hydro::ID], smallr);
  const auto ul = qL[Hydro::IU];
  const auto pl = fmax(qL[Hydro::IP], rl * smallp);
  const auto rr = fmax(qR[Hydro::ID], smallr);
  const auto ur = qR[Hydro::IU];
  const auto pr = fmax(qR[Hydro::IP], rr * smallp);

  // Lagrangian sound speed
  // const auto cl = gamma * pl * rl;
  // const auto cr = gamma * pr * rr;
  const auto ccl = eos.sound_speed(pl, rl);
  const auto cl = ccl * ccl * rl * rl;
  const auto ccr = eos.sound_speed(pr, rr);
  const auto cr = ccr * ccr * rr * rr;

  // First guess
  auto wl = sqrt(cl);
  auto wr = sqrt(cr);
  auto pstar = fmax(((wr * pl + wl * pr) + wl * wr * (ul - ur)) / (wl + wr), ZERO_F);
  auto pold = pstar;
  auto conv = ONE_F;

  // Newton-Raphson iterations to find pstar at the required accuracy
  for (int iter = 0; (iter < 10 /*niter_riemann*/) && (conv > KALYPSSO_NUM(1e-6)); ++iter)
  {
    const auto wwl = sqrt(cl * (ONE_F + gamma6 * (pold - pl) / pl));
    const auto wwr = sqrt(cr * (ONE_F + gamma6 * (pold - pr) / pr));
    const auto ql = KALYPSSO_NUM(2.0) * wwl * wwl * wwl / (wwl * wwl + cl);
    const auto qr = KALYPSSO_NUM(2.0) * wwr * wwr * wwr / (wwr * wwr + cr);
    const auto usl = ul - (pold - pl) / wwl;
    const auto usr = ur + (pold - pr) / wwr;
    const auto delp = fmax(qr * ql / (qr + ql) * (usl - usr), -pold);

    pold = pold + delp;
    conv = fabs(delp / (pold + smallpp)); // Convergence indicator
  }

  // Star region pressure
  // for a two-shock Riemann problem
  pstar = pold;
  wl = sqrt(cl * (ONE_F + gamma6 * (pstar - pl) / pl));
  wr = sqrt(cr * (ONE_F + gamma6 * (pstar - pr) / pr));

  // Star region velocity
  // for a two shock Riemann problem
  const auto ustar = HALF_F * (ul + (pl - pstar) / wl + ur - (pr - pstar) / wr);

  // Left going or right going contact wave
  real_t sgnm = COPYSIGN(ONE_F, ustar);

  // Left or right unperturbed state
  real_t ro, uo, po, wo;
  if (sgnm > ZERO_F)
  {
    ro = rl;
    uo = ul;
    po = pl;
    wo = wl;
  }
  else
  {
    ro = rr;
    uo = ur;
    po = pr;
    wo = wr;
  }

  const auto co = fmax(smallc, eos.sound_speed(po, ro));

  // Star region density (Shock, fmax prevents vacuum formation in star region)
  const auto rstar = fmax((ro / (ONE_F + ro * (po - pstar) / (wo * wo))), (smallr));

  // Star region sound speed
  const auto cstar = fmax(smallc, eos.sound_speed(pstar, rstar));

  // Compute rarefaction head and tail speed
  auto spout = co - sgnm * uo;
  auto spin = cstar - sgnm * ustar;
  // Compute shock speed
  const auto ushock = wo / ro - sgnm * uo;

  if (pstar >= po)
  {
    spin = ushock;
    spout = ushock;
  }

  // Sample the solution at x/t=0
  auto scr = fmax(spout - spin, smallc + fabs(spout + spin));
  auto frac = HALF_F * (ONE_F + (spout + spin) / scr);

  if (frac != frac) /* Not a Number */
    frac = KALYPSSO_NUM(0.0);
  else
    frac = frac >= KALYPSSO_NUM(1.0)   ? KALYPSSO_NUM(1.0)
           : frac <= KALYPSSO_NUM(0.0) ? KALYPSSO_NUM(0.0)
                                       : frac;

  qgdnv[Hydro::ID] = frac * rstar + (ONE_F - frac) * ro;
  qgdnv[Hydro::IU] = frac * ustar + (ONE_F - frac) * uo;
  qgdnv[Hydro::IP] = frac * pstar + (ONE_F - frac) * po;

  if (spout < ZERO_F)
  {
    qgdnv[Hydro::ID] = ro;
    qgdnv[Hydro::IU] = uo;
    qgdnv[Hydro::IP] = po;
  }

  if (spin > ZERO_F)
  {
    qgdnv[Hydro::ID] = rstar;
    qgdnv[Hydro::IU] = ustar;
    qgdnv[Hydro::IP] = pstar;
  }

  // transverse velocity
  if (sgnm > ZERO_F)
  {
    qgdnv[Hydro::IV] = qL[Hydro::IV];
    if constexpr (dim == 3)
      qgdnv[Hydro::IW] = qL[Hydro::IW];
  }
  else
  {
    qgdnv[Hydro::IV] = qR[Hydro::IV];
    if constexpr (dim == 3)
      qgdnv[Hydro::IW] = qR[Hydro::IW];
  }

  cmpflx<dim>(qgdnv, flux, eos);

} // riemann_approx

/**
 * Riemann solver, equivalent to riemann_llf in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * LLF = Local Lax-Friedrich (also called Rusanov flux).
 *
 * Reference : E.F. Toro, Riemann solvers and numerical methods for
 * fluid dynamics, Springer, chapter 10 (The HLL and HLLC Riemann solver).
 * See section 10.5.1, equation 10.43 which gives the expression of S+ denoted here
 * as cmax.
 *
 * @param[in] qL    : input left  state (primitive variables)
 * @param[in] qR    : input right state (primitive variables)
 * @param[out] flux : output flux
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_llf(HydroState<dim> const & qL,
            HydroState<dim> const & qR,
            HydroState<dim> &       flux,
            HydroSettings const &   settings,
            EosWrapper const &      eos)
{
  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // 1D LLF Riemann solver

  // constants
  auto const & smallr = settings.smallr;
  auto const & smallp = settings.smallp;

  //============================
  // Compute maximum wave speed
  //============================
  auto rl = fmax(qL[Hydro::ID], smallr);
  auto ul = qL[Hydro::IU];
  auto pl = fmax(qL[Hydro::IP], rl * smallp);

  auto rr = fmax(qR[Hydro::ID], smallr);
  auto ur = qR[Hydro::IU];
  auto pr = fmax(qR[Hydro::IP], rr * smallp);

  auto cl = eos.sound_speed(pl, rl);
  auto cr = eos.sound_speed(pr, rr);

  auto cmax = fmax(fabs(ul) + cl, fabs(ur) + cr);

  // Compute average velocity
  // qgdnv[Hydro::IU] = HALF_F*(qL[Hydro::IU]+qR[Hydro::IU]);

  //================================
  // Compute conservative variables
  //================================
  HydroState<dim> UL, UR;
  // mass density
  UL[Hydro::ID] = qL[Hydro::ID];
  UR[Hydro::ID] = qR[Hydro::ID];

  // total energy
  UL[Hydro::IP] = eos.volumic_eint_from_pressure(qL[Hydro::IP], qL[Hydro::ID]) +
                  HALF_F * qL[Hydro::ID] * qL[Hydro::IU] * qL[Hydro::IU];
  UR[Hydro::IP] = eos.volumic_eint_from_pressure(qR[Hydro::IP], qR[Hydro::ID]) +
                  HALF_F * qR[Hydro::ID] * qR[Hydro::IU] * qR[Hydro::IU];

  UL[Hydro::IP] += HALF_F * qL[Hydro::ID] * qL[Hydro::IV] * qL[Hydro::IV];
  UR[Hydro::IP] += HALF_F * qR[Hydro::ID] * qR[Hydro::IV] * qR[Hydro::IV];

  if constexpr (dim == 3)
  {
    UL[Hydro::IP] += HALF_F * qL[Hydro::ID] * qL[Hydro::IW] * qL[Hydro::IW];
    UR[Hydro::IP] += HALF_F * qR[Hydro::ID] * qR[Hydro::IW] * qR[Hydro::IW];
  }

  // normal momentum
  UL[Hydro::IU] = qL[Hydro::ID] * qL[Hydro::IU];
  UR[Hydro::IU] = qR[Hydro::ID] * qR[Hydro::IU];

  // transverse momentum
  UL[Hydro::IV] = qL[Hydro::ID] * qL[Hydro::IV];
  UR[Hydro::IV] = qR[Hydro::ID] * qR[Hydro::IV];

  if constexpr (dim == 3)
  {
    UL[Hydro::IW] = qL[Hydro::ID] * qL[Hydro::IW];
    UR[Hydro::IW] = qR[Hydro::ID] * qR[Hydro::IW];
  }

  //===============================
  // Compute left and right fluxes
  //===============================
  HydroState<dim> FL, FR;
  // mass density
  FL[Hydro::ID] = UL[Hydro::ID] * qL[Hydro::IU];
  FR[Hydro::ID] = UR[Hydro::ID] * qR[Hydro::IU];

  // total energy
  FL[Hydro::IP] = qL[Hydro::IU] * (UL[Hydro::IP] + qL[Hydro::IP]);
  FR[Hydro::IP] = qR[Hydro::IU] * (UR[Hydro::IP] + qR[Hydro::IP]);

  // normal momentum
  FL[Hydro::IU] = qL[Hydro::IP] + UL[Hydro::IU] * qL[Hydro::IU];
  FR[Hydro::IU] = qR[Hydro::IP] + UR[Hydro::IU] * qR[Hydro::IU];

  // transverse momentum
  FL[Hydro::IV] = FL[Hydro::ID] * qL[Hydro::IV];
  FR[Hydro::IV] = FR[Hydro::ID] * qR[Hydro::IV];

  if constexpr (dim == 3)
  {
    FL[Hydro::IW] = FL[Hydro::ID] * qL[Hydro::IW];
    FR[Hydro::IW] = FR[Hydro::ID] * qR[Hydro::IW];
  }

  //==============================
  // Compute Lax-Friedrich fluxes
  //==============================
  flux = HALF_F * (FL + FR - cmax * (UR - UL));

} // riemann_llf

/**
 * Riemann solver, equivalent to riemann_hll in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 *
 * This is the HYDRO only version. The MHD version is in file riemann_mhd.h
 *
 * Reference : E.F. Toro, Riemann solvers and numerical methods for
 * fluid dynamics, Springer, chapter 10 (The HLL and HLLC Riemann solver).
 *
 * @param[in] qL     : input left  state (primitive variables)
 * @param[in] qR     : input right state (primitive variables)
 * @param[out] flux  : output flux
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_hll(HydroState<dim> const & qL,
            HydroState<dim> const & qR,
            HydroState<dim> &       flux,
            HydroSettings const &   settings,
            EosWrapper const &      eos)
{

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // 1D HLL Riemann solver

  // constants
  auto const & smallr = settings.smallr;
  auto const & smallp = settings.smallp;

  // Maximum wave speed
  const auto rl = fmax(qL[Hydro::ID], smallr);
  const auto ul = qL[Hydro::IU];
  const auto pl = fmax(qL[Hydro::IP], rl * smallp);

  const auto rr = fmax(qR[Hydro::ID], smallr);
  const auto ur = qR[Hydro::IU];
  const auto pr = fmax(qR[Hydro::IP], rr * smallp);

  const auto cl = eos.sound_speed(pl, rl);
  const auto cr = eos.sound_speed(pr, rr);

  const auto SL = fmin(fmin(ul, ur) - fmax(cl, cr), ZERO_F);
  const auto SR = fmax(fmax(ul, ur) + fmax(cl, cr), ZERO_F);

  // Compute average velocity
  // qgdnv[Hydro::IU] = HALF_F*(qL[Hydro::IU]+qR[Hydro::IU]);

  // Compute conservative variables
  HydroState<dim> UL, UR;
  UL[Hydro::ID] = qL[Hydro::ID];
  UR[Hydro::ID] = qR[Hydro::ID];
  UL[Hydro::IP] = eos.volumic_eint_from_pressure(qL[Hydro::IP], qL[Hydro::ID]) +
                  HALF_F * qL[Hydro::ID] * qL[Hydro::IU] * qL[Hydro::IU];
  UR[Hydro::IP] = eos.volumic_eint_from_pressure(qR[Hydro::IP], qR[Hydro::ID]) +
                  HALF_F * qR[Hydro::ID] * qR[Hydro::IU] * qR[Hydro::IU];
  UL[Hydro::IP] += HALF_F * qL[Hydro::ID] * qL[Hydro::IV] * qL[Hydro::IV];
  UR[Hydro::IP] += HALF_F * qR[Hydro::ID] * qR[Hydro::IV] * qR[Hydro::IV];
  if constexpr (dim == 3)
  {
    UL[Hydro::IP] += HALF_F * qL[Hydro::ID] * qL[Hydro::IW] * qL[Hydro::IW];
    UR[Hydro::IP] += HALF_F * qR[Hydro::ID] * qR[Hydro::IW] * qR[Hydro::IW];
  }
  UL[Hydro::IU] = qL[Hydro::ID] * qL[Hydro::IU];
  UR[Hydro::IU] = qR[Hydro::ID] * qR[Hydro::IU];

  // Other advected quantities
  UL[Hydro::IV] = qL[Hydro::ID] * qL[Hydro::IV];
  UR[Hydro::IV] = qR[Hydro::ID] * qR[Hydro::IV];
  if constexpr (dim == 3)
  {
    UL[Hydro::IW] = qL[Hydro::ID] * qL[Hydro::IW];
    UR[Hydro::IW] = qR[Hydro::ID] * qR[Hydro::IW];
  }

  // Compute left and right fluxes
  HydroState<dim> FL, FR;
  FL[Hydro::ID] = UL[Hydro::IU];
  FR[Hydro::ID] = UR[Hydro::IU];
  FL[Hydro::IP] = qL[Hydro::IU] * (UL[Hydro::IP] + qL[Hydro::IP]);
  FR[Hydro::IP] = qR[Hydro::IU] * (UR[Hydro::IP] + qR[Hydro::IP]);
  FL[Hydro::IU] = qL[Hydro::IP] + UL[Hydro::IU] * qL[Hydro::IU];
  FR[Hydro::IU] = qR[Hydro::IP] + UR[Hydro::IU] * qR[Hydro::IU];

  // Other advected quantities
  FL[Hydro::IV] = FL[Hydro::ID] * qL[Hydro::IV];
  FR[Hydro::IV] = FR[Hydro::ID] * qR[Hydro::IV];
  if constexpr (dim == 3)
  {
    FL[Hydro::IW] = FL[Hydro::ID] * qL[Hydro::IW];
    FR[Hydro::IW] = FR[Hydro::ID] * qR[Hydro::IW];
  }

  // Compute HLL fluxes
  flux = (SR * FL - SL * FR + SR * SL * (UR - UL)) / (SR - SL);

} // riemann_hll

/**
 * Riemann solver HLLC
 *
 * @param[in] qL    : input left  state (primitive variables)
 * @param[in] qR    : input right state (primitive variables)
 * @param[out] flux : output flux
 *
 * Reference:
 * - Riemann Solvers and Numerical Methods for Fluid Dynamics. A practical introduction. E. F. Toro,
 * Springer (2009). https://link.springer.com/book/10.1007/b79761, chapter 10
 * - "On the Choice of Wavespeeds for the HLLC Riemann Solver", Batten et al., SIAM J. Sci. Comp.,
 * vol 18, issue 6, 1997, https://doi.org/10.1137/S1064827593260140
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_hllc(HydroState<dim> const & qL,
             HydroState<dim> const & qR,
             HydroState<dim> &       flux,
             HydroSettings const &   settings,
             EosWrapper const &      eos)
{
  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // UNUSED(qgdnv);

  auto const & smallr = settings.smallr;
  auto const & smallp = settings.smallp;
  auto const & smallc = settings.smallc;

  // Left variables
  const auto rl = fmax(qL[Hydro::ID], smallr);
  const auto pl = fmax(qL[Hydro::IP], rl * smallp);
  const auto ul = qL[Hydro::IU];

  auto ecinl = HALF_F * rl * ul * ul;
  ecinl += HALF_F * rl * qL[Hydro::IV] * qL[Hydro::IV];
  if constexpr (dim == 3)
    ecinl += HALF_F * rl * qL[Hydro::IW] * qL[Hydro::IW];

  const auto etotl = eos.volumic_eint_from_pressure(pl, rl) + ecinl;
  const auto ptotl = pl;

  // Right variables
  const auto rr = fmax(qR[Hydro::ID], smallr);
  const auto pr = fmax(qR[Hydro::IP], rr * smallp);
  const auto ur = qR[Hydro::IU];

  real_t ecinr = HALF_F * rr * ur * ur;
  ecinr += HALF_F * rr * qR[Hydro::IV] * qR[Hydro::IV];
  if constexpr (dim == 3)
    ecinr += HALF_F * rr * qR[Hydro::IW] * qR[Hydro::IW];

  const auto etotr = eos.volumic_eint_from_pressure(pr, rr) + ecinr;
  const auto ptotr = pr;

  // Find the largest eigenvalues in the normal direction to the interface
  const auto cfastl = fmax(eos.sound_speed(pl, rl), smallc);
  const auto cfastr = fmax(eos.sound_speed(pr, rr), smallc);

  // Compute HLL wave speed
  const auto SL = fmin(ul, ur) - fmax(cfastl, cfastr);
  const auto SR = fmax(ul, ur) + fmax(cfastl, cfastr);

  // Compute lagrangian sound speed
  const auto rcl = rl * (ul - SL);
  const auto rcr = rr * (SR - ur);

  // Compute acoustic star state
  const auto ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  const auto ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // Left star region variables
  const auto rstarl = rl * (SL - ul) / (SL - ustar);
  const auto etotstarl = ((SL - ul) * etotl - ptotl * ul + ptotstar * ustar) / (SL - ustar);

  // Right star region variables
  const auto rstarr = rr * (SR - ur) / (SR - ustar);
  const auto etotstarr = ((SR - ur) * etotr - ptotr * ur + ptotstar * ustar) / (SR - ustar);

  // Sample the solution at x/t=0
  real_t ro, uo, ptoto, etoto;
  if (SL > ZERO_F)
  {
    ro = rl;
    uo = ul;
    ptoto = ptotl;
    etoto = etotl;
  }
  else if (ustar > ZERO_F)
  {
    ro = rstarl;
    uo = ustar;
    ptoto = ptotstar;
    etoto = etotstarl;
  }
  else if (SR > ZERO_F)
  {
    ro = rstarr;
    uo = ustar;
    ptoto = ptotstar;
    etoto = etotstarr;
  }
  else
  {
    ro = rr;
    uo = ur;
    ptoto = ptotr;
    etoto = etotr;
  }

  // Compute the Godunov flux
  flux[Hydro::ID] = ro * uo;
  flux[Hydro::IU] = ro * uo * uo + ptoto;
  flux[Hydro::IP] = (etoto + ptoto) * uo;
  if (flux[Hydro::ID] > ZERO_F)
  {
    flux[Hydro::IV] = flux[Hydro::ID] * qL[Hydro::IV];
  }
  else
  {
    flux[Hydro::IV] = flux[Hydro::ID] * qR[Hydro::IV];
  }

  if constexpr (dim == 3)
  {
    if (flux[Hydro::ID] > ZERO_F)
    {
      flux[Hydro::IW] = flux[Hydro::ID] * qL[Hydro::IW];
    }
    else
    {
      flux[Hydro::IW] = flux[Hydro::ID] * qR[Hydro::IW];
    }
  }

} // riemann_hllc

/**
 * Riemann solver HLLC-LM
 *
 * This variant of HLLC has a low Mach correction for removing carbuncle instability problem
 * that arise when there is a strong axis-aligned shock.
 *
 * @param[in] qL    : input left  state (primitive variables)
 * @param[in] qR    : input right state (primitive variables)
 * @param[out] flux : output flux
 *
 * References:
 * - A low dissipation method to cure the grid-aligned shock instability. Nico
 * Fleischmann and Stefan Adami and Xiangyu Y. Hu and Nikolaus A. Adams, Journal of Computational
 * Physics Volume 401, 15 January 2020, 109004.
 * https://doi.org/10.1016/j.jcp.2019.109004
 * - A shock-stable modification of the HLLC Riemann solver with reduced numerical dissipation, Nico
 * Fleischmann and Stefan Adami and Nikolaus A. Adams, Journal of Computational Physics
 * Volume 423, 15 December 2020, 109762.
 * https://doi.org/10.1016/j.jcp.2020.109762
 * - Development of an accurate and robust HLLC-type Riemann solver for all Mach number flows, Lijun
 * Hu, Kexin Zhu, Lielong Li, Communications in Nonlinear Science and Numerical Simulation
 * Volume 152, Part A, January 2026, 109178.
 * https://doi.org/10.1016/j.cnsns.2025.109178
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_hllc_lm(HydroState<dim> const & qL,
                HydroState<dim> const & qR,
                HydroState<dim> &       flux,
                HydroSettings const &   settings,
                EosWrapper const &      eos)
{
  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  auto const & smallr = settings.smallr;
  auto const & smallp = settings.smallp;
  auto const & smallc = settings.smallc;


  HydroState<dim> UL, UR;

  //
  // Left variables
  //
  UL[Hydro::ID] = fmax(qL[Hydro::ID], smallr);
  auto &     rl = UL[Hydro::ID];
  const auto pl = fmax(qL[Hydro::IP], rl * smallp);
  UL[Hydro::IU] = rl * qL[Hydro::IU];
  auto const & ul = qL[Hydro::IU];

  auto ecinl = HALF_F * rl * ul * ul;
  ecinl += HALF_F * rl * qL[Hydro::IV] * qL[Hydro::IV];
  if constexpr (dim == 3)
    ecinl += HALF_F * rl * qL[Hydro::IW] * qL[Hydro::IW];

  UL[Hydro::IE] = eos.volumic_eint_from_pressure(pl, rl) + ecinl;

  UL[Hydro::IV] = rl * qL[Hydro::IV];
  if constexpr (dim == 3)
    UL[Hydro::IW] = rl * qL[Hydro::IW];

  //
  // Right variables
  //
  UR[Hydro::ID] = fmax(qR[Hydro::ID], smallr);
  auto &     rr = UR[Hydro::ID];
  const auto pr = fmax(qR[Hydro::IP], rr * smallp);
  UR[Hydro::IU] = rr * qR[Hydro::IU];
  auto const & ur = qR[Hydro::IU];

  auto ecinr = HALF_F * rr * ur * ur;
  ecinr += HALF_F * rr * qR[Hydro::IV] * qR[Hydro::IV];
  if constexpr (dim == 3)
    ecinr += HALF_F * rr * qR[Hydro::IW] * qR[Hydro::IW];

  UR[Hydro::IE] = eos.volumic_eint_from_pressure(pr, rr) + ecinr;

  UR[Hydro::IV] = rr * qR[Hydro::IV];
  if constexpr (dim == 3)
    UR[Hydro::IW] = rr * qR[Hydro::IW];

  // Find the largest eigenvalues in the normal direction to the interface
  const auto cfastl = fmax(eos.sound_speed(pl, rl), smallc);
  const auto cfastr = fmax(eos.sound_speed(pr, rr), smallc);

  // Compute HLL wave speed
  const auto SL = fmin(ul, ur) - fmax(cfastl, cfastr);
  const auto SR = fmax(ul, ur) + fmax(cfastl, cfastr);
  const auto Sstar =
    (pr - pl + rl * ul * (SL - ul) - rr * ur * (SR - ur)) / (rl * (SL - ul) - rr * (SR - ur));

  HydroState<dim> UstarL = UL;
  HydroState<dim> UstarR = UR;

  UstarL[Hydro::IU] = rl * Sstar;
  UstarL[Hydro::IE] = UL[Hydro::IE] + (Sstar - ul) * (rl * Sstar + pl / (SL - ul));
  UstarL *= (SL - ul) / (SL - Sstar);

  UstarR[Hydro::IU] = rr * Sstar;
  UstarR[Hydro::IE] = UR[Hydro::IE] + (Sstar - ur) * (rr * Sstar + pr / (SR - ur));
  UstarR *= (SR - ur) / (SR - Sstar);

  // Compute left and right fluxes
  HydroState<dim> FL, FR;
  FL[Hydro::ID] = UL[Hydro::IU];
  FR[Hydro::ID] = UR[Hydro::IU];
  FL[Hydro::IP] = qL[Hydro::IU] * (UL[Hydro::IP] + qL[Hydro::IP]);
  FR[Hydro::IP] = qR[Hydro::IU] * (UR[Hydro::IP] + qR[Hydro::IP]);
  FL[Hydro::IU] = qL[Hydro::IP] + UL[Hydro::IU] * qL[Hydro::IU];
  FR[Hydro::IU] = qR[Hydro::IP] + UR[Hydro::IU] * qR[Hydro::IU];

  // Other advected quantities
  FL[Hydro::IV] = FL[Hydro::ID] * qL[Hydro::IV];
  FR[Hydro::IV] = FR[Hydro::ID] * qR[Hydro::IV];
  if constexpr (dim == 3)
  {
    FL[Hydro::IW] = FL[Hydro::ID] * qL[Hydro::IW];
    FR[Hydro::IW] = FR[Hydro::ID] * qR[Hydro::IW];
  }

  if (SL >= 0)
  {
    flux = FL;
  }
  else if (SR <= 0)
  {
    flux = FR;
  }
  else
  {
    // sound speed
    const auto cl = eos.sound_speed(pl, rl);
    const auto cr = eos.sound_speed(pr, rr);

    // low Mach correction
    const auto Ma_lim = KALYPSSO_NUM(0.1);
    const auto Ma_loc = fmax(fabs(ul / cl), fabs(ur / cr));

    // low Mach correction
    const auto phi = sin(fmin(ONE_F, Ma_loc / Ma_lim) * PI_F * HALF_F);

    flux =
      HALF_F * (FL + FR) + HALF_F * (phi * SL * (UstarL - UL) + fabs(Sstar) * (UstarL - UstarR) +
                                     phi * SR * (UstarR - UR));
  }

} // riemann_hllc_lm

/**
 * Wrapper function calling the actual riemann solver.
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION void
riemann_hydro(HydroState<dim> const & qL,
              HydroState<dim> const & qR,
              HydroState<dim> &       flux,
              HydroSettings const &   settings,
              EosWrapper const &      eos)
{

  if (settings.riemannSolverType == +RiemannSolverType::APPROX)
  {
    riemann_approx<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLL)
  {
    riemann_hll<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLLC)
  {
    riemann_hllc<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLLC_LM)
  {
    riemann_hllc_lm<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::LLF)
  {
    riemann_llf<dim>(qL, qR, flux, settings, eos);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type");
  }

} // riemann_hydro

/**
 * Another wrapper function calling the actual riemann solver.
 */
template <size_t dim, typename EosWrapper>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       riemann_hydro(HydroState<dim> const & qL,
                                     HydroState<dim> const & qR,
                                     HydroSettings const &   settings,
                                     EosWrapper const &      eos)
{

  HydroState<dim> flux;

  if (settings.riemannSolverType == +RiemannSolverType::APPROX)
  {
    riemann_approx<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLL)
  {
    riemann_hll<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLLC)
  {
    riemann_hllc<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::HLLC_LM)
  {
    riemann_hllc_lm<dim>(qL, qR, flux, settings, eos);
  }
  else if (settings.riemannSolverType == +RiemannSolverType::LLF)
  {
    riemann_llf<dim>(qL, qR, flux, settings, eos);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type");
  }

  return flux;

} // riemann_hydro

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_H_
