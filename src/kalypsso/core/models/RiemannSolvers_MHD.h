// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RiemannSolvers_MHD.h
 * All possible Riemann solvers or so for MHD.
 */
#ifndef KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_MHD_H_
#define KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_MHD_H_

#include <math.h>

#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/models/MHDState.h>
#include <kalypsso/core/models/MHDSettings.h>
#include <kalypsso/core/models/mhd_utils.h>
#include <kalypsso/core/models/riemann_solver_types.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

using MagneticField_t = Kokkos::Array<real_t, 3>;

// ====================================================================
// ====================================================================
/**
 * MHD HLL Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * \param[in] qleft  : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 *
 */
KOKKOS_INLINE_FUNCTION
void
riemann_hll(MHDStateCell &      qleft,
            MHDStateCell &      qright,
            MHDStateCell &      flux,
            MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // enforce continuity of normal component
  real_t bx_mean = HALF_F * (qleft[MHD::IA] + qright[MHD::IA]);

  qleft[MHD::IA] = bx_mean;
  qright[MHD::IA] = bx_mean;

  MHDStateCell uleft, fleft;
  MHDStateCell uright, fright;

  core::models::mhd::find_mhd_flux(qleft, uleft, fleft, settings);
  core::models::mhd::find_mhd_flux(qright, uright, fright, settings);

  // find the largest eigenvalue in the normal direction to the interface
  real_t cfleft = core::models::mhd::find_speed_fast<IX>(qleft, settings);
  real_t cfright = core::models::mhd::find_speed_fast<IX>(qright, settings);

  real_t vleft = qleft[MHD::IU];
  real_t vright = qright[MHD::IU];
  real_t sl = fmin(fmin(vleft, vright) - fmax(cfleft, cfright), ZERO_F);
  real_t sr = fmax(fmax(vleft, vright) + fmax(cfleft, cfright), ZERO_F);

  // the hll flux
  flux[MHD::ID] =
    (sr * fleft[MHD::ID] - sl * fright[MHD::ID] + sr * sl * (uright[MHD::ID] - uleft[MHD::ID])) /
    (sr - sl);
  flux[MHD::IP] =
    (sr * fleft[MHD::IP] - sl * fright[MHD::IP] + sr * sl * (uright[MHD::IP] - uleft[MHD::IP])) /
    (sr - sl);
  flux[MHD::IU] =
    (sr * fleft[MHD::IU] - sl * fright[MHD::IU] + sr * sl * (uright[MHD::IU] - uleft[MHD::IU])) /
    (sr - sl);
  flux[MHD::IV] =
    (sr * fleft[MHD::IV] - sl * fright[MHD::IV] + sr * sl * (uright[MHD::IV] - uleft[MHD::IV])) /
    (sr - sl);
  flux[MHD::IW] =
    (sr * fleft[MHD::IW] - sl * fright[MHD::IW] + sr * sl * (uright[MHD::IW] - uleft[MHD::IW])) /
    (sr - sl);
  flux[MHD::IA] =
    (sr * fleft[MHD::IA] - sl * fright[MHD::IA] + sr * sl * (uright[MHD::IA] - uleft[MHD::IA])) /
    (sr - sl);
  flux[MHD::IB] =
    (sr * fleft[MHD::IB] - sl * fright[MHD::IB] + sr * sl * (uright[MHD::IB] - uleft[MHD::IB])) /
    (sr - sl);
  flux[MHD::IC] =
    (sr * fleft[MHD::IC] - sl * fright[MHD::IC] + sr * sl * (uright[MHD::IC] - uleft[MHD::IC])) /
    (sr - sl);

} // riemann_hll

// ====================================================================
// ====================================================================
/*
 * MHD LLF (Local Lax-Friedrich) Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * The following code is adapted from Dumses.
 *
 * \param[in] qleft  : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] settings : hydro settings
 *
 */
KOKKOS_INLINE_FUNCTION
void
riemann_llf(MHDStateCell &      qleft,
            MHDStateCell &      qright,
            MHDStateCell &      flux,
            MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // enforce continuity of normal component
  real_t bx_mean = HALF_F * (qleft[MHD::IA] + qright[MHD::IA]);
  qleft[MHD::IA] = bx_mean;
  qright[MHD::IA] = bx_mean;

  MHDStateCell uleft, fleft;
  MHDStateCell uright, fright;

  core::models::mhd::find_mhd_flux(qleft, uleft, fleft, settings);
  core::models::mhd::find_mhd_flux(qright, uright, fright, settings);

  // compute mean flux
  for (uint32_t iVar = 0; iVar < MHDStateCell::size(); iVar++)
    flux[iVar] = (fleft[iVar] + fright[iVar]) / 2;

  // find the largest eigenvalue in the normal direction to the interface
  real_t cleft = core::models::mhd::find_speed_info(qleft, settings);
  real_t cright = core::models::mhd::find_speed_info(qright, settings);

  real_t vel_info = fmax(cleft, cright);

  // the Local Lax-Friedrich flux
  for (uint32_t iVar = 0; iVar < MHDStateCell::size(); iVar++)
    flux[iVar] -= vel_info * (uright[iVar] - uleft[iVar]) / 2;

} // riemann_llf

// ====================================================================
// ====================================================================
/**
 * Riemann solver HLLC
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] flux  : output flux
 */
KOKKOS_INLINE_FUNCTION void
riemann_hllc(MHDStateCell const & qleft,
             MHDStateCell const & qright,
             MHDStateCell &       flux,
             MHDSettings const &  settings)
{
  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // UNUSED(qgdnv);

  real_t const & gamma0 = settings.hydro.gamma0;
  real_t const & smallr = settings.hydro.smallr;
  real_t const & smallp = settings.hydro.smallp;
  real_t const & smallc = settings.hydro.smallc;

  real_t const entho = ONE_F / (gamma0 - ONE_F);

  // Left variables
  real_t rl = fmax(qleft[MHD::ID], smallr);
  real_t pl = fmax(qleft[MHD::IP], rl * smallp);
  real_t ul = qleft[MHD::IU];

  real_t ecinl = HALF_F * rl * ul * ul;
  ecinl += HALF_F * rl * qleft[MHD::IV] * qleft[MHD::IV];
  ecinl += HALF_F * rl * qleft[MHD::IW] * qleft[MHD::IW];

  real_t etotl = pl * entho + ecinl;
  real_t ptotl = pl;

  // Right variables
  real_t rr = fmax(qright[MHD::ID], smallr);
  real_t pr = fmax(qright[MHD::IP], rr * smallp);
  real_t ur = qright[MHD::IU];

  real_t ecinr = HALF_F * rr * ur * ur;
  ecinr += HALF_F * rr * qright[MHD::IV] * qright[MHD::IV];
  ecinr += HALF_F * rr * qright[MHD::IW] * qright[MHD::IW];

  real_t etotr = pr * entho + ecinr;
  real_t ptotr = pr;

  // Find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = sqrt(fmax(gamma0 * pl / rl, smallc * smallc));
  real_t cfastr = sqrt(fmax(gamma0 * pr / rr, smallc * smallc));

  // Compute HLL wave speed
  real_t SL = fmin(ul, ur) - fmax(cfastl, cfastr);
  real_t SR = fmax(ul, ur) + fmax(cfastl, cfastr);

  // Compute lagrangian sound speed
  real_t rcl = rl * (ul - SL);
  real_t rcr = rr * (SR - ur);

  // Compute acoustic star state
  real_t ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  real_t ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // Left star region variables
  real_t rstarl = rl * (SL - ul) / (SL - ustar);
  real_t etotstarl = ((SL - ul) * etotl - ptotl * ul + ptotstar * ustar) / (SL - ustar);

  // Right star region variables
  real_t rstarr = rr * (SR - ur) / (SR - ustar);
  real_t etotstarr = ((SR - ur) * etotr - ptotr * ur + ptotstar * ustar) / (SR - ustar);

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
  flux[MHD::ID] = ro * uo;
  flux[MHD::IU] = ro * uo * uo + ptoto;
  flux[MHD::IP] = (etoto + ptoto) * uo;
  if (flux[MHD::ID] > ZERO_F)
  {
    flux[MHD::IV] = flux[MHD::ID] * qleft[MHD::IV];
  }
  else
  {
    flux[MHD::IV] = flux[MHD::ID] * qright[MHD::IV];
  }

  {
    if (flux[MHD::ID] > ZERO_F)
    {
      flux[MHD::IW] = flux[MHD::ID] * qleft[MHD::IW];
    }
    else
    {
      flux[MHD::IW] = flux[MHD::ID] * qright[MHD::IW];
    }
  }

} // riemann_hllc

// ====================================================================
// ====================================================================
/**
 * Riemann solver, equivalent to riemann_hlld in RAMSES/DUMSES (see file
 * godunov_utils.f90 in RAMSES/DUMSES).
 *
 * Reference :
 * <A
 * HREF="http://www.sciencedirect.com/science/article/B6WHY-4FY3P80-7/2/426234268c96dcca8a828d098b75fe4e">
 * Miyoshi & Kusano, 2005, JCP, 208, 315 </A>
 *
 * \warning This version of HLLD integrates the pressure term in
 * flux[MHD::IU] (as in RAMSES). This will need to be modified in the
 * future (as it is done in DUMSES) to handle cylindrical / spherical
 * coordinate systems. For example, one could add a new output named qStar
 * to store star state, and that could be used to compute geometrical terms
 * outside this routine.
 *
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 */
KOKKOS_INLINE_FUNCTION
void
riemann_hlld(MHDStateCell &      qleft,
             MHDStateCell &      qright,
             MHDStateCell &      flux,
             MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // Constants
  const real_t gamma0 = settings.hydro.gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  // Enforce continuity of normal component of magnetic field
  real_t a = HALF_F * (qleft[MHD::IA] + qright[MHD::IA]);
  real_t sgnm = (a >= 0) ? ONE_F : -ONE_F;

  qleft[MHD::IA] = a;
  qright[MHD::IA] = a;

  // ISOTHERMAL
  real_t cIso = settings.hydro.cIso;
  if (cIso > 0)
  {
    // recompute pressure
    qleft[MHD::IP] = qleft[MHD::ID] * cIso * cIso;
    qright[MHD::IP] = qright[MHD::ID] * cIso * cIso;
  } // end ISOTHERMAL

  // left variables
  real_t rl, pl, ul, vl, wl, bl, cl;
  rl = qleft[MHD::ID]; // rl = fmax(qleft[MHD::ID], static_cast<real_t>(gParams.smallr)    );
  pl = qleft[MHD::IP]; // pl = fmax(qleft[MHD::IP], static_cast<real_t>(rl*gParams.smallp) );
  ul = qleft[MHD::IU];
  vl = qleft[MHD::IV];
  wl = qleft[MHD::IW];
  bl = qleft[MHD::IB];
  cl = qleft[MHD::IC];
  real_t ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  real_t emagl = HALF_F * (a * a + bl * bl + cl * cl);
  real_t etotl = pl * entho + ecinl + emagl;
  real_t ptotl = pl + emagl;
  real_t vdotbl = ul * a + vl * bl + wl * cl;

  // right variables
  real_t rr, pr, ur, vr, wr, br, cr;
  rr = qright[MHD::ID]; // rr = fmax(qright[MHD::ID], static_cast<real_t>( gParams.smallr) );
  pr = qright[MHD::IP]; // pr = fmax(qright[MHD::IP], static_cast<real_t>( rr*gParams.smallp) );
  ur = qright[MHD::IU];
  vr = qright[MHD::IV];
  wr = qright[MHD::IW];
  br = qright[MHD::IB];
  cr = qright[MHD::IC];
  real_t ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  real_t emagr = HALF_F * (a * a + br * br + cr * cr);
  real_t etotr = pr * entho + ecinr + emagr;
  real_t ptotr = pr + emagr;
  real_t vdotbr = ur * a + vr * br + wr * cr;

  // find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = core::models::mhd::find_speed_fast<IX>(qleft, settings);
  real_t cfastr = core::models::mhd::find_speed_fast<IX>(qright, settings);

  // compute hll wave speed
  real_t sl = fmin(ul, ur) - fmax(cfastl, cfastr);
  real_t sr = fmax(ul, ur) + fmax(cfastl, cfastr);

  // compute lagrangian sound speed
  real_t rcl = rl * (ul - sl);
  real_t rcr = rr * (sr - ur);

  // compute acoustic star state
  real_t ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  real_t ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // left star region variables
  real_t estar;
  real_t rstarl, el;
  rstarl = rl * (sl - ul) / (sl - ustar);
  estar = rl * (sl - ul) * (sl - ustar) - a * a;
  el = rl * (sl - ul) * (sl - ul) - a * a;
  real_t vstarl, wstarl;
  real_t bstarl, cstarl;

  if (fabs(estar) < KALYPSSO_NUM(1e-4) * (a * a))
  {
    vstarl = vl;
    bstarl = bl;
    wstarl = wl;
    cstarl = cl;
  }
  else
  {
    vstarl = vl - a * bl * (ustar - ul) / estar;
    bstarl = bl * el / estar;
    wstarl = wl - a * cl * (ustar - ul) / estar;
    cstarl = cl * el / estar;
  }
  real_t vdotbstarl = ustar * a + vstarl * bstarl + wstarl * cstarl;
  real_t etotstarl =
    ((sl - ul) * etotl - ptotl * ul + ptotstar * ustar + a * (vdotbl - vdotbstarl)) / (sl - ustar);
  real_t sqrrstarl = sqrt(rstarl);
  real_t calfvenl = fabs(a) / sqrrstarl; /* sqrrstarl should never be zero, but it might happen if
                                            border conditions are not OK !!!!!! */
  real_t sal = ustar - calfvenl;

  // right star region variables
  real_t rstarr, er;
  rstarr = rr * (sr - ur) / (sr - ustar);
  estar = rr * (sr - ur) * (sr - ustar) - a * a;
  er = rr * (sr - ur) * (sr - ur) - a * a;
  real_t vstarr, wstarr;
  real_t bstarr, cstarr;

  if (fabs(estar) < KALYPSSO_NUM(1e-4) * (a * a))
  {
    vstarr = vr;
    bstarr = br;
    wstarr = wr;
    cstarr = cr;
  }
  else
  {
    vstarr = vr - a * br * (ustar - ur) / estar;
    bstarr = br * er / estar;
    wstarr = wr - a * cr * (ustar - ur) / estar;
    cstarr = cr * er / estar;
  }
  real_t vdotbstarr = ustar * a + vstarr * bstarr + wstarr * cstarr;
  real_t etotstarr =
    ((sr - ur) * etotr - ptotr * ur + ptotstar * ustar + a * (vdotbr - vdotbstarr)) / (sr - ustar);
  real_t sqrrstarr = sqrt(rstarr);
  real_t calfvenr = fabs(a) / sqrrstarr; /* sqrrstarr should never be zero, but it might happen if
                                            border conditions are not OK !!!!!! */
  real_t sar = ustar + calfvenr;

  // double star region variables
  real_t vstarstar =
    (sqrrstarl * vstarl + sqrrstarr * vstarr + sgnm * (bstarr - bstarl)) / (sqrrstarl + sqrrstarr);
  real_t wstarstar =
    (sqrrstarl * wstarl + sqrrstarr * wstarr + sgnm * (cstarr - cstarl)) / (sqrrstarl + sqrrstarr);
  real_t bstarstar =
    (sqrrstarl * bstarr + sqrrstarr * bstarl + sgnm * sqrrstarl * sqrrstarr * (vstarr - vstarl)) /
    (sqrrstarl + sqrrstarr);
  real_t cstarstar =
    (sqrrstarl * cstarr + sqrrstarr * cstarl + sgnm * sqrrstarl * sqrrstarr * (wstarr - wstarl)) /
    (sqrrstarl + sqrrstarr);
  real_t vdotbstarstar = ustar * a + vstarstar * bstarstar + wstarstar * cstarstar;
  real_t etotstarstarl = etotstarl - sgnm * sqrrstarl * (vdotbstarl - vdotbstarstar);
  real_t etotstarstarr = etotstarr + sgnm * sqrrstarr * (vdotbstarr - vdotbstarstar);

  // sample the solution at x/t=0
  real_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
  if (sl > 0)
  { // flow is supersonic, return upwind variables
    ro = rl;
    uo = ul;
    vo = vl;
    wo = wl;
    bo = bl;
    co = cl;
    ptoto = ptotl;
    etoto = etotl;
    vdotbo = vdotbl;
  }
  else if (sal > 0)
  {
    ro = rstarl;
    uo = ustar;
    vo = vstarl;
    wo = wstarl;
    bo = bstarl;
    co = cstarl;
    ptoto = ptotstar;
    etoto = etotstarl;
    vdotbo = vdotbstarl;
  }
  else if (ustar > 0)
  {
    ro = rstarl;
    uo = ustar;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstar;
    etoto = etotstarstarl;
    vdotbo = vdotbstarstar;
  }
  else if (sar > 0)
  {
    ro = rstarr;
    uo = ustar;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstar;
    etoto = etotstarstarr;
    vdotbo = vdotbstarstar;
  }
  else if (sr > 0)
  {
    ro = rstarr;
    uo = ustar;
    vo = vstarr;
    wo = wstarr;
    bo = bstarr;
    co = cstarr;
    ptoto = ptotstar;
    etoto = etotstarr;
    vdotbo = vdotbstarr;
  }
  else
  { // flow is supersonic, return upwind variables
    ro = rr;
    uo = ur;
    vo = vr;
    wo = wr;
    bo = br;
    co = cr;
    ptoto = ptotr;
    etoto = etotr;
    vdotbo = vdotbr;
  }

  // compute the godunov flux
  flux[MHD::ID] = ro * uo;
  flux[MHD::IP] = (etoto + ptoto) * uo - a * vdotbo;
  flux[MHD::IU] =
    ro * uo * uo - a * a +
    ptoto; /* *** WARNING *** : ptoto used here (this is only valid for cartesian geometry) ! */
  flux[MHD::IV] = ro * uo * vo - a * bo;
  flux[MHD::IW] = ro * uo * wo - a * co;
  flux[MHD::IA] = ZERO_F;
  flux[MHD::IB] = bo * uo - a * vo;
  flux[MHD::IC] = co * uo - a * wo;

} // riemann_hlld

// ====================================================================
// ====================================================================
/**
 * Riemann solver 5 waves
 *
 * Reference :
 * https://www.sciencedirect.com/science/article/pii/S0021999124007034
 * Tremblin, Pascal, et al. "A multi-dimensional, robust, and cell-centered finite-volume scheme for
 * the ideal MHD equations. " Journal of Computational Physics 519 (2024): 113455.
 *
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 */
KOKKOS_INLINE_FUNCTION
void
riemann_5_waves(MHDStateCell &      qleft,
                MHDStateCell &      qright,
                MHDStateCell &      flux,
                MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;
  // Constants
  const real_t gamma0 = settings.hydro.gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  // left variables
  real_t rl, pl, ul, vl, wl, al, bl, cl;
  rl = qleft[MHD::ID];
  pl = qleft[MHD::IP];
  ul = qleft[MHD::IU];
  vl = qleft[MHD::IV];
  wl = qleft[MHD::IW];
  al = qleft[MHD::IA];
  bl = qleft[MHD::IB];
  cl = qleft[MHD::IC];
  real_t ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  real_t emagl = HALF_F * (al * al + bl * bl + cl * cl);
  real_t etotl = pl * entho + ecinl + emagl;
  real_t ptotl = pl + emagl;
  real_t cs2l = gamma0 * (pl / rl); // square sound speed


  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxL, cRelaxL;

  pRelaxL[IX] = ptotl - al * al;
  pRelaxL[IY] = -al * bl;
  pRelaxL[IZ] = -al * cl;

  cRelaxL[IX] = sqrt(rl * (rl * cs2l + ((bl * bl) + (cl * cl) + emagl)));
  cRelaxL[IY] = sqrt(rl * ((al * al) + emagl)) + 1e-14;
  cRelaxL[IZ] = sqrt(rl * ((al * al) + emagl)) + 1e-14;


  // right variables
  real_t rr, pr, ur, vr, wr, ar, br, cr;
  rr = qright[MHD::ID];
  pr = qright[MHD::IP];
  ur = qright[MHD::IU];
  vr = qright[MHD::IV];
  wr = qright[MHD::IW];
  ar = qright[MHD::IA];
  br = qright[MHD::IB];
  cr = qright[MHD::IC];
  real_t ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  real_t emagr = HALF_F * (ar * ar + br * br + cr * cr);
  real_t etotr = pr * entho + ecinr + emagr;
  real_t ptotr = pr + emagr;
  real_t cs2r = gamma0 * (pr / rr); // square sound speed

  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxR, cRelaxR;

  pRelaxR[IX] = ptotr - ar * ar;
  pRelaxR[IY] = -ar * br;
  pRelaxR[IZ] = -ar * cr;

  cRelaxR[IX] = sqrt(rr * (rr * cs2r + ((br * br) + (cr * cr) + emagr)));
  cRelaxR[IY] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;
  cRelaxR[IZ] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;

  // Calculation of ustar and pstar
  Kokkos::Array<real_t, 3> ustar, pstar;

  ustar[IX] =
    (cRelaxL[IX] * ul + cRelaxR[IX] * ur + pRelaxL[IX] - pRelaxR[IX]) / (cRelaxL[IX] + cRelaxR[IX]);
  ustar[IY] =
    (cRelaxL[IY] * vl + cRelaxR[IY] * vr + pRelaxL[IY] - pRelaxR[IY]) / (cRelaxL[IY] + cRelaxR[IY]);
  ustar[IZ] =
    (cRelaxL[IZ] * wl + cRelaxR[IZ] * wr + pRelaxL[IZ] - pRelaxR[IZ]) / (cRelaxL[IZ] + cRelaxR[IZ]);

  // const real_t Ma_norm = fabs(ustar[IX])/fmin(sqrt(cs2l), sqrt(cs2r));

  // const real_t theta_norm = ONE_F;//fmin(Ma_norm, ONE_F);

  pstar[IX] = (cRelaxR[IX] * pRelaxL[IX] + cRelaxL[IX] * pRelaxR[IX] +
               cRelaxL[IX] * cRelaxR[IX] * (ul - ur)) /
              (cRelaxL[IX] + cRelaxR[IX]);
  pstar[IY] = (cRelaxR[IY] * pRelaxL[IY] + cRelaxL[IY] * pRelaxR[IY] +
               cRelaxL[IY] * cRelaxR[IY] * (vl - vr)) /
              (cRelaxL[IY] + cRelaxR[IY]);
  pstar[IZ] = (cRelaxR[IZ] * pRelaxL[IZ] + cRelaxL[IZ] * pRelaxR[IZ] +
               cRelaxL[IZ] * cRelaxR[IZ] * (wl - wr)) /
              (cRelaxL[IZ] + cRelaxR[IZ]);


  real_t pdotu = ustar[IX] * pstar[IX] + ustar[IY] * pstar[IY] + ustar[IZ] * pstar[IZ];

  MHDStateCell qStencil;
  real_t       B_next;
  real_t       E_next;

  if (ustar[IX] > ZERO_F)
  {
    qStencil = qleft;
    B_next = ar;
    E_next = etotl;
  }
  else
  {
    qStencil = qright;
    B_next = al;
    E_next = etotr;
  }

  // compute the godunov flux
  flux[MHD::ID] = qStencil[MHD::ID] * ustar[IX];
  flux[MHD::IU] = qStencil[MHD::ID] * qStencil[MHD::IU] * ustar[IX] + pstar[IX];
  flux[MHD::IV] = qStencil[MHD::ID] * qStencil[MHD::IV] * ustar[IX] + pstar[IY];
  flux[MHD::IW] = qStencil[MHD::ID] * qStencil[MHD::IW] * ustar[IX] + pstar[IZ];
  flux[MHD::IA] = qStencil[MHD::IA] * ustar[IX] - B_next * ustar[IX];
  flux[MHD::IB] = qStencil[MHD::IB] * ustar[IX] - B_next * ustar[IY];
  flux[MHD::IC] = qStencil[MHD::IC] * ustar[IX] - B_next * ustar[IZ];
  flux[MHD::IP] = E_next * ustar[IX] + pdotu;


} // riemann_5_waves

// ====================================================================
// ====================================================================
/**
 * Riemann solver 5 waves with entropy correction
 *
 * Reference :
 * https://www.sciencedirect.com/science/article/pii/S0021999124007034
 * Tremblin, Pascal, et al. "A multi-dimensional, robust, and cell-centered finite-volume scheme for
 * the ideal MHD equations. " Journal of Computational Physics 519 (2024): 113455.
 *
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 */
KOKKOS_INLINE_FUNCTION
void
riemann_5_waves_entropy(MHDStateCell &      qleft,
                        MHDStateCell &      qright,
                        MHDStateCell &      flux,
                        MHDSettings const & settings)
{
  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;
  // Constants
  const real_t gamma0 = settings.hydro.gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  // left variables
  real_t rl, pl, ul, vl, wl, al, bl, cl;
  rl = qleft[MHD::ID];
  pl = qleft[MHD::IP];
  ul = qleft[MHD::IU];
  vl = qleft[MHD::IV];
  wl = qleft[MHD::IW];
  al = qleft[MHD::IA];
  bl = qleft[MHD::IB];
  cl = qleft[MHD::IC];
  real_t ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  real_t emagl = HALF_F * (al * al + bl * bl + cl * cl);
  real_t etotl = pl * entho + ecinl + emagl;
  real_t ptotl = pl + emagl;
  real_t cs2l = gamma0 * (pl / rl); // square sound speed


  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxL, cRelaxL;

  pRelaxL[IX] = ptotl - (al * al);
  pRelaxL[IY] = -(al * bl);
  pRelaxL[IZ] = -(al * cl);

  cRelaxL[IX] = sqrt(rl * (rl * cs2l + ((bl * bl) + (cl * cl) + emagl)));
  cRelaxL[IY] = sqrt(rl * ((al * al) + emagl)) + 1e-14;
  cRelaxL[IZ] = sqrt(rl * ((al * al) + emagl)) + 1e-14;


  // right variables
  real_t rr, pr, ur, vr, wr, ar, br, cr;
  rr = qright[MHD::ID];
  pr = qright[MHD::IP];
  ur = qright[MHD::IU];
  vr = qright[MHD::IV];
  wr = qright[MHD::IW];
  ar = qright[MHD::IA];
  br = qright[MHD::IB];
  cr = qright[MHD::IC];
  real_t ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  real_t emagr = HALF_F * (ar * ar + br * br + cr * cr);
  real_t etotr = pr * entho + ecinr + emagr;
  real_t ptotr = pr + emagr;
  real_t cs2r = gamma0 * (pr / rr); // square sound speed

  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxR, cRelaxR;

  pRelaxR[IX] = ptotr - (ar * ar);
  pRelaxR[IY] = -(ar * br);
  pRelaxR[IZ] = -(ar * cr);

  cRelaxR[IX] = sqrt(rr * (rr * cs2r + ((br * br) + (cr * cr) + emagr)));
  cRelaxR[IY] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;
  cRelaxR[IZ] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;

  // Calculation of ustar and pstar
  Kokkos::Array<real_t, 3> ustar, pstar;

  ustar[IX] =
    (cRelaxL[IX] * ul + cRelaxR[IX] * ur + pRelaxL[IX] - pRelaxR[IX]) / (cRelaxL[IX] + cRelaxR[IX]);
  ustar[IY] =
    (cRelaxL[IY] * vl + cRelaxR[IY] * vr + pRelaxL[IY] - pRelaxR[IY]) / (cRelaxL[IY] + cRelaxR[IY]);
  ustar[IZ] =
    (cRelaxL[IZ] * wl + cRelaxR[IZ] * wr + pRelaxL[IZ] - pRelaxR[IZ]) / (cRelaxL[IZ] + cRelaxR[IZ]);

  // const real_t Ma_norm = fabs(ustar[IX])/fmin(sqrt(cs2l), sqrt(cs2r));

  // const real_t theta_norm = ONE_F;//fmin(Ma_norm, ONE_F);

  pstar[IX] = (cRelaxR[IX] * pRelaxL[IX] + cRelaxL[IX] * pRelaxR[IX] +
               cRelaxL[IX] * cRelaxR[IX] * (ul - ur)) /
              (cRelaxL[IX] + cRelaxR[IX]);
  pstar[IY] = (cRelaxR[IY] * pRelaxL[IY] + cRelaxL[IY] * pRelaxR[IY] +
               cRelaxL[IY] * cRelaxR[IY] * (vl - vr)) /
              (cRelaxL[IY] + cRelaxR[IY]);
  pstar[IZ] = (cRelaxR[IZ] * pRelaxL[IZ] + cRelaxL[IZ] * pRelaxR[IZ] +
               cRelaxL[IZ] * cRelaxR[IZ] * (wl - wr)) /
              (cRelaxL[IZ] + cRelaxR[IZ]);


  real_t pdotu = ustar[IX] * pstar[IX] + ustar[IY] * pstar[IY] + ustar[IZ] * pstar[IZ];

  MHDStateCell qStencil;
  real_t       B_next;
  real_t       E_next;

  if (ustar[IX] > ZERO_F)
  {
    qStencil = qleft;
    B_next = ar;
    E_next = etotl;
  }
  else
  {
    qStencil = qright;
    B_next = al;
    E_next = etotr;
  }

  // auto beta = pr / emagr;
  // auto alfven = sqrt(abs(ecinr) / abs(emagr));
  const auto beta = fmin(pl / emagl, pr / emagr);
  const auto alfven = fmax(sqrt(ecinl / emagl), sqrt(ecinr / emagr));

  if (beta < settings.small_beta || alfven > settings.large_alfven)
  {

    // compute the godunov flux
    flux[MHD::ID] = qStencil[MHD::ID] * ustar[IX];
    flux[MHD::IU] = qStencil[MHD::ID] * qStencil[MHD::IU] * ustar[IX] + pstar[IX];
    flux[MHD::IV] = qStencil[MHD::ID] * qStencil[MHD::IV] * ustar[IX] + pstar[IY];
    flux[MHD::IW] = qStencil[MHD::ID] * qStencil[MHD::IW] * ustar[IX] + pstar[IZ];
    flux[MHD::IA] = qStencil[MHD::IA] * ustar[IX];
    flux[MHD::IB] = qStencil[MHD::IB] * ustar[IX];
    flux[MHD::IC] = qStencil[MHD::IC] * ustar[IX];
    flux[MHD::IP] = E_next * ustar[IX] + pdotu;
  }
  else
  {
    // compute the godunov flux
    flux[MHD::ID] = qStencil[MHD::ID] * ustar[IX];
    flux[MHD::IU] = qStencil[MHD::ID] * qStencil[MHD::IU] * ustar[IX] + pstar[IX];
    flux[MHD::IV] = qStencil[MHD::ID] * qStencil[MHD::IV] * ustar[IX] + pstar[IY];
    flux[MHD::IW] = qStencil[MHD::ID] * qStencil[MHD::IW] * ustar[IX] + pstar[IZ];
    flux[MHD::IA] = qStencil[MHD::IA] * ustar[IX] - B_next * ustar[IX];
    flux[MHD::IB] = qStencil[MHD::IB] * ustar[IX] - B_next * ustar[IY];
    flux[MHD::IC] = qStencil[MHD::IC] * ustar[IX] - B_next * ustar[IZ];
    flux[MHD::IP] = E_next * ustar[IX] + pdotu;
  }
} // riemann_5_waves_entropy

// ====================================================================
// ====================================================================
/**
 * Riemann solver, HLLD with Boris correction
 *
 * Reference : [Matsumoto2019]
 * <A
 * HREF="https://iopscience.iop.org/article/10.3847/1538-4357/ab05cb/meta">
 * Matsumoto, T., Miyoshi, T., & Takasao, S. (2019). </A>
 *
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 */
KOKKOS_INLINE_FUNCTION
void
riemann_hlld_boris(MHDStateCell &      qleft,
                   MHDStateCell &      qright,
                   MHDStateCell &      flux,
                   MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // Constants
  const real_t gamma0 = settings.hydro.gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);
  const real_t cboris2 = settings.cboris * settings.cboris;

  // Enforce continuity of normal component of magnetic field
  real_t a = HALF_F * (qleft[MHD::IA] + qright[MHD::IA]);
  real_t sgnm = (a >= 0) ? ONE_F : -ONE_F;

  qleft[MHD::IA] = a;
  qright[MHD::IA] = a;

  // ISOTHERMAL
  real_t cIso = settings.hydro.cIso;
  if (cIso > 0)
  {
    // recompute pressure
    qleft[MHD::IP] = qleft[MHD::ID] * cIso * cIso;
    qright[MHD::IP] = qright[MHD::ID] * cIso * cIso;
  } // end ISOTHERMAL

  // left variables
  real_t rl, pl, ul, vl, wl, bl, cl, ga2l;
  ga2l = core::models::mhd::compute_Boris(qleft, settings);
  rl = qleft[MHD::ID]; // rl = fmax(qleft[Hydro::ID], static_cast<real_t>(gParams.smallr)    );
  pl = qleft[MHD::IP]; // pl = fmax(qleft[Hydro::IP], static_cast<real_t>(rl*gParams.smallp) );
  ul = qleft[MHD::IU] * ga2l;
  vl = qleft[MHD::IV] * ga2l;
  wl = qleft[MHD::IW] * ga2l;
  bl = qleft[MHD::IB];
  cl = qleft[MHD::IC];

  real_t ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  real_t emagl = HALF_F * (a * a + bl * bl + cl * cl);
  real_t etotl = pl * entho + ecinl + emagl;
  real_t ptotl = pl + emagl;
  real_t vdotbl = ul * a + vl * bl + wl * cl;

  // right variables
  real_t rr, pr, ur, vr, wr, br, cr, ga2r;
  ga2r = core::models::mhd::compute_Boris(qright, settings);
  rr = qright[MHD::ID]; // rr = fmax(qright[Hydro::ID], static_cast<real_t>( gParams.smallr) );
  pr = qright[MHD::IP]; // pr = fmax(qright[Hydro::IP], static_cast<real_t>( rr*gParams.smallp) );
  ur = qright[MHD::IU] * ga2r;
  vr = qright[MHD::IV] * ga2r;
  wr = qright[MHD::IW] * ga2r;
  br = qright[MHD::IB];
  cr = qright[MHD::IC];
  real_t ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  real_t emagr = HALF_F * (a * a + br * br + cr * cr);
  real_t etotr = pr * entho + ecinr + emagr;
  real_t ptotr = pr + emagr;
  real_t vdotbr = ur * a + vr * br + wr * cr;

  // find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = core::models::mhd::find_speed_fast_boris<IX>(qleft, settings);
  real_t cfastr = core::models::mhd::find_speed_fast_boris<IX>(qright, settings);

  // compute hll wave speed
  real_t sl = fmin(ul, ur) - fmax(cfastl, cfastr);
  real_t sr = fmax(ul, ur) + fmax(cfastl, cfastr);

  // HLL average used to calculate Sm equation (41) in [Matsumoto2019]
  real_t rhll = ((sr * rr) - (sl * rl) - (rr * ur) + (rl * ul)) / (sr - sl);
  real_t ahll = a;

  real_t Fbl = ul * bl - vl * a;
  real_t Fbr = ur * br - vr * a;
  real_t Fcl = ul * cl - wl * a;
  real_t Fcr = ur * cr - wr * a;

  real_t bhll = ((sr * br) - (sl * bl) - Fbr + Fbl) / (sr - sl);
  real_t chll = ((sr * cr) - (sl * cl) - Fcr + Fcl) / (sr - sl);
  real_t Vahll = ((ahll * ahll) + (bhll * bhll) + (chll * chll)) / rhll;

  real_t ga2hll = ONE_F / (ONE_F + Vahll / cboris2);

  // compute acoustic star state
  real_t sm = (((sr / ga2r - ur) * rr * ur - (sl / ga2l - ul) * rl * ul - ptotr + ptotl) * ga2hll) /
              ((sr - ur) * rr - (sl - ul) * rl);

  real_t ptotstarl =
    ptotl - (sl / ga2l - ul) * rl * ul + (sl / ga2l - sm) * rl * sm * (sl - ul) / (sl - sm);
  real_t ptotstarr =
    ptotr - (sr / ga2r - ur) * rr * ur + (sr / ga2r - sm) * rr * sm * (sr - ur) / (sr - sm);

  // left star region variable
  real_t rstarl = rl * (sl - ul) / (sl - sm);
  real_t estar, estarup, el;
  estar = rl * (sl - ul) * (sl / ga2l - sm) - a * a;
  estarup = rl * (sl / ga2l - ul) * (sl - sm) - a * a;
  el = rl * (sl - ul) * (sl - ul) * (sl / ga2l - sm) / (sl - sm) - a * a;

  real_t vstarl, wstarl;
  real_t bstarl, cstarl;
  if (fabs(estar) < KALYPSSO_NUM(1e-4) * (a * a))
  {
    vstarl = vl;
    bstarl = bl;
    wstarl = wl;
    cstarl = cl;
  }
  else
  {
    vstarl = (estarup * vl - a * bl * (sm - ul)) / estar;
    wstarl = (estarup * wl - a * cl * (sm - ul)) / estar;
    bstarl = bl * el / estar + (ONE_F - estarup / estar) * vl * a / (sl - sm);
    cstarl = cl * el / estar + (ONE_F - estarup / estar) * wl * a / (sl - sm);
  }
  real_t vdotbstarl = sm * a + vstarl * bstarl + wstarl * cstarl;
  real_t etotstarl =
    ((sl - ul) * etotl - ptotl * ul + ptotstarl * sm + a * (vdotbl - vdotbstarl)) / (sl - sm);
  // real_t sqrrstarl = sqrt(ga2l / rstarl);
  // real_t calfvenl = fabs(a) / sqrrstarl;
  real_t sal = sm - fabs(a) * sqrt(ga2l / rstarl);


  // right star region variable
  real_t rstarr = rr * (sr - ur) / (sr - sm);
  real_t er;
  estar = rr * (sr - ur) * (sr / ga2r - sm) - a * a;
  estarup = rr * (sr / ga2r - ur) * (sr - sm) - a * a;
  er = rr * (sr - ur) * (sr - ur) * (sr / ga2r - sm) / (sr - sm) - a * a;

  real_t vstarr, wstarr;
  real_t bstarr, cstarr;
  if (fabs(estar) < KALYPSSO_NUM(1e-4) * (a * a))
  {
    vstarr = vr;
    bstarr = br;
    wstarr = wr;
    cstarr = cr;
  }
  else
  {
    vstarr = (estarup * vr - a * br * (sm - ur)) / estar;
    wstarr = (estarup * wr - a * cr * (sm - ur)) / estar;
    bstarr = br * er / estar + (ONE_F - estarup / estar) * vr * a / (sr - sm);
    cstarr = cr * er / estar + (ONE_F - estarup / estar) * wr * a / (sr - sm);
  }
  real_t vdotbstarr = sm * a + vstarr * bstarr + wstarr * cstarr;
  real_t etotstarr =
    ((sr - ur) * etotr - ptotr * ur + ptotstarr * sm + a * (vdotbr - vdotbstarr)) / (sr - sm);
  // real_t sqrrstarl = sqrt(ga2l / rstarl);
  // real_t calfvenl = fabs(a) / sqrrstarl;

  real_t sar = sm + fabs(a) * sqrt(ga2r / rstarr);

  // double star region variables
  real_t vstarstar =
    (sqrt(rstarl / ga2l) * vstarl + sqrt(rstarr / ga2r) * vstarr + sgnm * (bstarr - bstarl) +
     (ONE_F - ONE_F / ga2l) * sm * rstarl * vstarl / fabs(a) -
     (ONE_F - ONE_F / ga2r) * sm * rstarr * vstarr / fabs(a)) /
    (sqrt(rstarl / ga2l) + sqrt(rstarr / ga2r));

  real_t wstarstar =
    (sqrt(rstarl / ga2l) * wstarl + sqrt(rstarr / ga2r) * wstarr + sgnm * (cstarr - cstarl) +
     (ONE_F - ONE_F / ga2l) * sm * rstarl * wstarl / fabs(a) -
     (ONE_F - ONE_F / ga2r) * sm * rstarr * wstarr / fabs(a)) /
    (sqrt(rstarl / ga2l) + sqrt(rstarr / ga2r));

  real_t bstarstar = (sqrt(ga2r * rstarl) * bstarr + sqrt(ga2l * rstarr) * bstarl +
                      sqrt(rstarl * rstarr) * (vstarr - vstarl) * sgnm) /
                     (sqrt(ga2r * rstarl) + sqrt(ga2l * rstarr));

  real_t cstarstar = (sqrt(ga2r * rstarl) * cstarr + sqrt(ga2l * rstarr) * cstarl +
                      sqrt(rstarl * rstarr) * (wstarr - wstarl) * sgnm) /
                     (sqrt(ga2r * rstarl) + sqrt(ga2l * rstarr));

  real_t vdotbstarstar = sm * a + vstarstar * bstarstar + wstarstar * cstarstar;

  real_t etotstarstarl = etotstarl - sgnm * sqrt(rstarl / ga2l) * (vdotbstarl - vdotbstarstar);
  real_t etotstarstarr = etotstarr + sgnm * sqrt(rstarr / ga2r) * (vdotbstarr - vdotbstarstar);

  // sample the solution at x/t=0
  real_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
  if (sl > 0)
  { // flow is supersonic, return upwind variables
    ro = rl;
    uo = ul;
    vo = vl;
    wo = wl;
    bo = bl;
    co = cl;
    ptoto = ptotl;
    etoto = etotl;
    vdotbo = vdotbl;
  }
  else if (sal > 0)
  {
    ro = rstarl;
    uo = sm;
    vo = vstarl;
    wo = wstarl;
    bo = bstarl;
    co = cstarl;
    ptoto = ptotstarl;
    etoto = etotstarl;
    vdotbo = vdotbstarl;
  }
  else if (sm > 0)
  {
    ro = rstarl;
    uo = sm;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstarl;
    etoto = etotstarstarl;
    vdotbo = vdotbstarstar;
  }
  else if (sar > 0)
  {
    ro = rstarr;
    uo = sm;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstarr;
    etoto = etotstarstarr;
    vdotbo = vdotbstarstar;
  }
  else if (sr > 0)
  {
    ro = rstarr;
    uo = sm;
    vo = vstarr;
    wo = wstarr;
    bo = bstarr;
    co = cstarr;
    ptoto = ptotstarr;
    etoto = etotstarr;
    vdotbo = vdotbstarr;
  }
  else
  { // flow is supersonic, return upwind variables
    ro = rr;
    uo = ur;
    vo = vr;
    wo = wr;
    bo = br;
    co = cr;
    ptoto = ptotr;
    etoto = etotr;
    vdotbo = vdotbr;
  }

  // compute the godunov flux
  flux[MHD::ID] = ro * uo;
  flux[MHD::IP] = (etoto + ptoto) * uo - a * vdotbo;
  flux[MHD::IU] = ro * uo * uo - a * a + ptoto;
  flux[MHD::IV] = ro * uo * vo - a * bo;
  flux[MHD::IW] = ro * uo * wo - a * co;
  flux[MHD::IA] = ZERO_F;
  flux[MHD::IB] = bo * uo - a * vo;
  flux[MHD::IC] = co * uo - a * wo;

} // riemann_hlld_boris

// ====================================================================
// ====================================================================
/**
 * Wrapper function calling the actual riemann solver for MHD.
 */
KOKKOS_INLINE_FUNCTION
void
riemann_mhd(MHDStateCell &      qleft,
            MHDStateCell &      qright,
            MHDStateCell &      flux,
            MHDSettings const & settings)
{

  if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLD)
  {
    riemann_hlld(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLL)
  {
    riemann_hll(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::LLF)
  {
    riemann_llf(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLC)
  {
    // only useful for debug
    riemann_hllc(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::FIVE_WAVES)
  {
    riemann_5_waves(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLD_BORIS)
  {
    riemann_hlld_boris(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::FIVE_WAVES_ENTROPY)
  {
    riemann_5_waves_entropy(qleft, qright, flux, settings);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type");
  }

} // riemann_mhd

/**
 * Another wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
MHDStateCell
riemann_mhd(MHDStateCell & qleft, MHDStateCell & qright, MHDSettings const & settings)
{
  MHDStateCell flux;

  if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLD)
  {
    riemann_hlld(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLL)
  {
    riemann_hll(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::LLF)
  {
    riemann_llf(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLC)
  {
    // only useful for debug
    riemann_hllc(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::FIVE_WAVES)
  {
    riemann_5_waves(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLD_BORIS)
  {
    riemann_hlld_boris(qleft, qright, flux, settings);
  }
  else if (settings.hydro.riemannSolverType == +RiemannSolverType::FIVE_WAVES_ENTROPY)
  {
    riemann_5_waves_entropy(qleft, qright, flux, settings);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type");
  }

  return flux;
} // riemann_mhd

// ====================================================================
// =================== CASE SPLIT MAGNETIC FIELD ======================
// ====================================================================

/**
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[in] MFL    : input left state zero order magnetic field
 * \param[in] MFR    : input right state zero order magnetic field
 * \param[out] flux  : output flux
 * \param[in] params : hydro parameters
 */

KOKKOS_INLINE_FUNCTION
void
riemann_hlld(MHDSplitStateCell & qleft,
             MHDSplitStateCell & qright,
             MagneticField_t &   MFL,
             MagneticField_t &   MFR,
             MHDStateCell &      flux,
             MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // Constants
  const real_t gamma0 = settings.hydro.gamma0;
  const real_t entho = ONE_F / (gamma0 - ONE_F);

  real_t a0, b0, c0;

  a0 = HALF_F * (MFL[IX] + MFR[IX]);
  b0 = HALF_F * (MFL[IY] + MFR[IY]);
  c0 = HALF_F * (MFL[IZ] + MFR[IZ]);

  // Enforce continuity of normal component of magnetic field
  real_t a = HALF_F * (qleft[MHD::IA] + qright[MHD::IA]);

  // KALYPSSO_INFO("VALEUR DE A0={} ET DE A1={}",a0, a);

  real_t sgnm = ((a + a0) >= 0) ? ONE_F : -ONE_F;

  // KALYPSSO_INFO("VALEUR DE A = {} ET sgnm = {} ", a, sgnm);

  qleft[MHD::IA] = a;
  qright[MHD::IA] = a;

  // ISOTHERMAL
  real_t cIso = settings.hydro.cIso;
  if (cIso > 0)
  {
    // recompute pressure
    qleft[MHD::IP] = qleft[MHD::ID] * cIso * cIso;
    qright[MHD::IP] = qright[MHD::ID] * cIso * cIso;
  } // end ISOTHERMAL

  // left variables
  real_t rl, pl, ul, vl, wl, bl, cl;
  rl = qleft[MHD::ID]; // rl = fmax(qleft[MHD::ID], static_cast<real_t>(gParams.smallr)    );
  pl = qleft[MHD::IP]; // pl = fmax(qleft[MHD::IP], static_cast<real_t>(rl*gParams.smallp) );
  ul = qleft[MHD::IU];
  vl = qleft[MHD::IV];
  wl = qleft[MHD::IW];
  bl = qleft[MHD::IB];
  cl = qleft[MHD::IC];

  real_t ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  real_t emagl = HALF_F * ((a * a) + (bl * bl) + (cl * cl));
  real_t etotl = pl * entho + ecinl + emagl;
  real_t vdotbl = (ul * a) + (vl * bl) + (wl * cl);
  // Eq. 7 from [Miyoshi2010] for total pressure
  real_t ptotl = pl + emagl + ((a * a0) + (bl * b0) + (cl * c0));


  // right variables
  real_t rr, pr, ur, vr, wr, br, cr;
  rr = qright[MHD::ID]; // rr = fmax(qright[MHD::ID], static_cast<real_t>( gParams.smallr) );
  pr = qright[MHD::IP]; // pr = fmax(qright[MHD::IP], static_cast<real_t>( rr*gParams.smallp) );
  ur = qright[MHD::IU];
  vr = qright[MHD::IV];
  wr = qright[MHD::IW];
  br = qright[MHD::IB];
  cr = qright[MHD::IC];

  real_t ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  real_t emagr = HALF_F * ((a * a) + (br * br) + (cr * cr));
  real_t etotr = pr * entho + ecinr + emagr;
  real_t vdotbr = (ur * a) + (vr * br) + (wr * cr);
  // Eq. 7 from [Miyoshi2010] for total pressure
  real_t ptotr = pr + emagr + ((a * a0) + (br * b0) + (cr * c0));

  // find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = core::models::mhd::find_speed_fast<IX>(qleft, MFL, settings);
  real_t cfastr = core::models::mhd::find_speed_fast<IX>(qright, MFR, settings);

  // compute hll wave speed
  real_t sl = fmin(ul, ur) - fmax(cfastl, cfastr);
  real_t sr = fmax(ul, ur) + fmax(cfastl, cfastr);

  // compute lagrangian sound speed
  real_t rcl = rl * (ul - sl);
  real_t rcr = rr * (sr - ur);

  // compute acoustic star state
  real_t ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  real_t ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // left star region variables
  real_t estarl;
  real_t rstarl, el;
  // clang-format off
  rstarl = rl * (sl - ul) / (sl - ustar);
  estarl = rl * (sl - ul) * (sl - ustar) - ((a + a0) * (a + a0));
  el     = rl * (sl - ul) * (sl - ul)    - ((a + a0) * (a + a0));
  // clang-format on
  real_t vstarl, wstarl;
  real_t bstarl, cstarl;

  // if (fabs(estarl) < KALYPSSO_NUM(1e-4) * ((a + a0) * (a + a0)))
  if (fabs(estarl) < KALYPSSO_NUM(1e-99))
  {
    vstarl = vl;
    bstarl = -b0; // In [Miyoshi2010] after equation 44
    wstarl = wl;
    cstarl = -c0; // In [Miyoshi2010] after equation 44
  }
  else
  {
    vstarl = vl - ((a + a0) * (bl + b0) * (ustar - ul) / estarl);
    bstarl = ((bl + b0) * (el / estarl)) - b0;
    wstarl = wl - ((a + a0) * (cl + c0) * (ustar - ul) / estarl);
    cstarl = ((cl + c0) * (el / estarl)) - c0;
  }
  real_t vdotbstarl = ustar * a + vstarl * bstarl + wstarl * cstarl;
  real_t etotstarl =
    ((sl - ul) * etotl - ptotl * ul + ptotstar * ustar + (a + a0) * (vdotbl - vdotbstarl)) /
    (sl - ustar);
  real_t sqrrstarl = sqrt(rstarl);
  real_t calfvenl =
    fabs((a + a0)) / sqrrstarl; // sqrrstarl should never be zero, but it might happen if
                                //  border conditions are not OK !!!!!!
  real_t sal = ustar - calfvenl;

  // right star region variables
  real_t rstarr, er, estarr;
  // clang-format off
  rstarr = rr * (sr - ur) / (sr - ustar);
  estarr = rr * (sr - ur) * (sr - ustar) - ((a + a0) * (a + a0));
  er     = rr * (sr - ur) * (sr - ur)    - ((a + a0) * (a + a0));
  // clang-format on
  real_t vstarr, wstarr;
  real_t bstarr, cstarr;

  // if (fabs(estarr) < KALYPSSO_NUM(1e-4) * ((a + a0) * (a + a0)))
  if (fabs(estarr) < KALYPSSO_NUM(1e-99))
  {
    vstarr = vr;
    bstarr = -b0; // In [Miyoshi2010] after equation 44
    wstarr = wr;
    cstarr = -c0; // In [Miyoshi2010] after equation 44
  }
  else
  {
    vstarr = vr - ((a + a0) * (br + b0) * (ustar - ur) / estarr);
    bstarr = ((br + b0) * (er / estarr)) - b0;
    wstarr = wr - ((a + a0) * (cr + c0) * (ustar - ur) / estarr);
    cstarr = ((cr + c0) * (er / estarr)) - c0;
  }
  real_t vdotbstarr = ustar * a + vstarr * bstarr + wstarr * cstarr;
  real_t etotstarr =
    ((sr - ur) * etotr - ptotr * ur + ptotstar * ustar + (a + a0) * (vdotbr - vdotbstarr)) /
    (sr - ustar);
  real_t sqrrstarr = sqrt(rstarr);
  real_t calfvenr = fabs((a + a0)) / sqrrstarr; // sqrrstarr should never be zero, but it might
                                                // happen if border conditions are not OK !!!!!!
  real_t sar = ustar + calfvenr;

  // double star region variables
  real_t vstarstar =
    (sqrrstarl * vstarl + sqrrstarr * vstarr + sgnm * (bstarr - bstarl)) / (sqrrstarl + sqrrstarr);
  real_t wstarstar =
    (sqrrstarl * wstarl + sqrrstarr * wstarr + sgnm * (cstarr - cstarl)) / (sqrrstarl + sqrrstarr);
  real_t bstarstar =
    (sqrrstarl * bstarr + sqrrstarr * bstarl + sgnm * sqrrstarl * sqrrstarr * (vstarr - vstarl)) /
    (sqrrstarl + sqrrstarr);
  real_t cstarstar =
    (sqrrstarl * cstarr + sqrrstarr * cstarl + sgnm * sqrrstarl * sqrrstarr * (wstarr - wstarl)) /
    (sqrrstarl + sqrrstarr);
  real_t vdotbstarstar = ustar * a + vstarstar * bstarstar + wstarstar * cstarstar;
  real_t etotstarstarl = etotstarl - sgnm * sqrrstarl * (vdotbstarl - vdotbstarstar);
  real_t etotstarstarr = etotstarr + sgnm * sqrrstarr * (vdotbstarr - vdotbstarstar);

  // sample the solution at x/t=0
  real_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
  if (sl > 0)
  { // flow is supersonic, return upwind variables
    ro = rl;
    uo = ul;
    vo = vl;
    wo = wl;
    bo = bl;
    co = cl;
    ptoto = ptotl;
    etoto = etotl;
    vdotbo = vdotbl;
  }
  else if (sal > 0)
  {
    ro = rstarl;
    uo = ustar;
    vo = vstarl;
    wo = wstarl;
    bo = bstarl;
    co = cstarl;
    ptoto = ptotstar;
    etoto = etotstarl;
    vdotbo = vdotbstarl;
  }
  else if (ustar > 0)
  {
    ro = rstarl;
    uo = ustar;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstar;
    etoto = etotstarstarl;
    vdotbo = vdotbstarstar;
  }
  else if (sar > 0)
  {
    ro = rstarr;
    uo = ustar;
    vo = vstarstar;
    wo = wstarstar;
    bo = bstarstar;
    co = cstarstar;
    ptoto = ptotstar;
    etoto = etotstarstarr;
    vdotbo = vdotbstarstar;
  }
  else if (sr > 0)
  {
    ro = rstarr;
    uo = ustar;
    vo = vstarr;
    wo = wstarr;
    bo = bstarr;
    co = cstarr;
    ptoto = ptotstar;
    etoto = etotstarr;
    vdotbo = vdotbstarr;
  }
  else
  { // flow is supersonic, return upwind variables
    ro = rr;
    uo = ur;
    vo = vr;
    wo = wr;
    bo = br;
    co = cr;
    ptoto = ptotr;
    etoto = etotr;
    vdotbo = vdotbr;
  }

  // compute the godunov flux

  // clang-format off
  flux[MHD::ID] = ro * uo;
  flux[MHD::IP] = (etoto + ptoto) * uo - (a + a0) * vdotbo;
  flux[MHD::IU] = ro * uo * uo + ptoto - (a + a0) * a  - (a * a0);
  flux[MHD::IV] = ro * uo * vo         - (a + a0) * bo - (a * b0);
  flux[MHD::IW] = ro * uo * wo         - (a + a0) * co - (a * c0);
  flux[MHD::IA] = ZERO_F;
  flux[MHD::IB] = (bo + b0) * uo - (a + a0) * vo;
  flux[MHD::IC] = (co + c0) * uo - (a + a0) * wo;
  // clang-format on

} // riemann_hlld

KOKKOS_INLINE_FUNCTION
MHDStateCell
riemann_mhd(MHDSplitStateCell & qleft,
            MHDSplitStateCell & qright,
            MagneticField_t &   MFL,
            MagneticField_t &   MFR,
            MHDSettings const & settings)
{
  MHDStateCell flux;

  if (settings.hydro.riemannSolverType == +RiemannSolverType::HLLD)
  {
    riemann_hlld(qleft, qright, MFL, MFR, flux, settings);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type");
  }

  return flux;
} // riemann_mhd

/**
 * Riemann solver 5 waves : compute ustar
 *
 * Reference :
 * https://www.sciencedirect.com/science/article/pii/S0021999124007034
 * Tremblin, Pascal, et al. "A multi-dimensional, robust, and cell-centered finite-volume scheme for
 * the ideal MHD equations. " Journal of Computational Physics 519 (2024): 113455.
 *
 * \param[in] qleft : input left state
 * \param[in] qright : input right state
 * \param[out] ustar  : output star velocity
 * \param[in] settings : MHD settings
 */
KOKKOS_INLINE_FUNCTION
Kokkos::Array<real_t, 3>
ustar_5_waves(MHDStateCell & qleft, MHDStateCell & qright, MHDSettings const & settings)
{
  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // Constants
  auto const & gamma0 = settings.hydro.gamma0;

  // left variables
  // real_t rl, pl, ul, vl, wl, al, bl, cl;
  auto const & rl = qleft[MHD::ID];
  auto const & pl = qleft[MHD::IP];
  auto const & ul = qleft[MHD::IU];
  auto const & vl = qleft[MHD::IV];
  auto const & wl = qleft[MHD::IW];
  auto const & al = qleft[MHD::IA];
  auto const & bl = qleft[MHD::IB];
  auto const & cl = qleft[MHD::IC];
  const auto   ecinl = HALF_F * (ul * ul + vl * vl + wl * wl) * rl;
  const auto   emagl = HALF_F * (al * al + bl * bl + cl * cl);
  const auto   ptotl = pl + emagl;
  const auto   cs2l = gamma0 * (pl / rl); // square sound speed

  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxL, cRelaxL;

  pRelaxL[IX] = ptotl - (al * al);
  pRelaxL[IY] = -(al * bl);
  pRelaxL[IZ] = -(al * cl);

  cRelaxL[IX] = sqrt(rl * (rl * cs2l + ((bl * bl) + (cl * cl) + emagl)));
  cRelaxL[IY] = sqrt(rl * ((al * al) + emagl)) + 1e-14;
  cRelaxL[IZ] = sqrt(rl * ((al * al) + emagl)) + 1e-14;

  // right variables
  // real_t rr, pr, ur, vr, wr, ar, br, cr;
  auto const & rr = qright[MHD::ID];
  auto const & pr = qright[MHD::IP];
  auto const & ur = qright[MHD::IU];
  auto const & vr = qright[MHD::IV];
  auto const & wr = qright[MHD::IW];
  auto const & ar = qright[MHD::IA];
  auto const & br = qright[MHD::IB];
  auto const & cr = qright[MHD::IC];
  const auto   ecinr = HALF_F * (ur * ur + vr * vr + wr * wr) * rr;
  const auto   emagr = HALF_F * (ar * ar + br * br + cr * cr);
  const auto   ptotr = pr + emagr;
  const auto   cs2r = gamma0 * (pr / rr); // square sound speed

  // Definition and calculation of the relaxed pressure and the relaxation parameters
  Kokkos::Array<real_t, 3> pRelaxR, cRelaxR;

  pRelaxR[IX] = ptotr - (ar * ar);
  pRelaxR[IY] = -(ar * br);
  pRelaxR[IZ] = -(ar * cr);

  cRelaxR[IX] = sqrt(rr * (rr * cs2r + ((br * br) + (cr * cr) + emagr)));
  cRelaxR[IY] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;
  cRelaxR[IZ] = sqrt(rr * ((ar * ar) + emagr)) + 1e-14;

  const auto beta = fmin(pl / emagl, pr / emagr);
  const auto alfven = fmax(sqrt(ecinl / emagl), sqrt(ecinr / emagr));

  Kokkos::Array<real_t, 3> ustar{ ZERO_F, ZERO_F, ZERO_F };

  if (beta < settings.small_beta || alfven > settings.large_alfven)
  {
    ustar[IX] = (cRelaxL[IX] * ul + cRelaxR[IX] * ur + pRelaxL[IX] - pRelaxR[IX]) /
                (cRelaxL[IX] + cRelaxR[IX]);
    ustar[IY] = (cRelaxL[IY] * vl + cRelaxR[IY] * vr + pRelaxL[IY] - pRelaxR[IY]) /
                (cRelaxL[IY] + cRelaxR[IY]);
    ustar[IZ] = (cRelaxL[IZ] * wl + cRelaxR[IZ] * wr + pRelaxL[IZ] - pRelaxR[IZ]) /
                (cRelaxL[IZ] + cRelaxR[IZ]);
  }

  return ustar;

} // ustar_5_waves

KOKKOS_INLINE_FUNCTION
Kokkos::Array<real_t, 3>
compute_ustar(MHDStateCell & qleft, MHDStateCell & qright, MHDSettings const & settings)
{
  return ustar_5_waves(qleft, qright, settings);
}

/**
 * 2D magnetic riemann solver of type HLLD
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
mag_riemann2d_hlld(const MHDStateCell (&qLLRR)[4], real_t eLLRR[4], MHDSettings const & settings)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // alias reference to input arrays
  const MHDStateCell & qLL = qLLRR[ILL];
  const MHDStateCell & qRL = qLLRR[IRL];
  const MHDStateCell & qLR = qLLRR[ILR];
  const MHDStateCell & qRR = qLLRR[IRR];

  real_t & ELL = eLLRR[ILL];
  real_t & ERL = eLLRR[IRL];
  real_t & ELR = eLLRR[ILR];
  real_t & ERR = eLLRR[IRR];
  // real_t ELL,ERL,ELR,ERR;

  const real_t & rLL = qLL[MHD::ID];
  const real_t & pLL = qLL[MHD::IP];
  const real_t & uLL = qLL[MHD::IU];
  const real_t & vLL = qLL[MHD::IV];
  const real_t & aLL = qLL[MHD::IA];
  const real_t & bLL = qLL[MHD::IB];
  const real_t & cLL = qLL[MHD::IC];

  const real_t & rLR = qLR[MHD::ID];
  const real_t & pLR = qLR[MHD::IP];
  const real_t & uLR = qLR[MHD::IU];
  const real_t & vLR = qLR[MHD::IV];
  const real_t & aLR = qLR[MHD::IA];
  const real_t & bLR = qLR[MHD::IB];
  const real_t & cLR = qLR[MHD::IC];

  const real_t & rRL = qRL[MHD::ID];
  const real_t & pRL = qRL[MHD::IP];
  const real_t & uRL = qRL[MHD::IU];
  const real_t & vRL = qRL[MHD::IV];
  const real_t & aRL = qRL[MHD::IA];
  const real_t & bRL = qRL[MHD::IB];
  const real_t & cRL = qRL[MHD::IC];

  const real_t & rRR = qRR[MHD::ID];
  const real_t & pRR = qRR[MHD::IP];
  const real_t & uRR = qRR[MHD::IU];
  const real_t & vRR = qRR[MHD::IV];
  const real_t & aRR = qRR[MHD::IA];
  const real_t & bRR = qRR[MHD::IB];
  const real_t & cRR = qRR[MHD::IC];

  // Compute 4 fast magnetosonic velocity relative to x direction
  real_t cFastLLx = core::models::mhd::find_speed_fast<IX>(qLL, settings);
  real_t cFastLRx = core::models::mhd::find_speed_fast<IX>(qLR, settings);
  real_t cFastRLx = core::models::mhd::find_speed_fast<IX>(qRL, settings);
  real_t cFastRRx = core::models::mhd::find_speed_fast<IX>(qRR, settings);

  // Compute 4 fast magnetosonic velocity relative to y direction
  real_t cFastLLy = core::models::mhd::find_speed_fast<IY>(qLL, settings);
  real_t cFastLRy = core::models::mhd::find_speed_fast<IY>(qLR, settings);
  real_t cFastRLy = core::models::mhd::find_speed_fast<IY>(qRL, settings);
  real_t cFastRRy = core::models::mhd::find_speed_fast<IY>(qRR, settings);

  // TODO : write a find_speed that computes the 2 speeds together (in
  // a single routine -> factorize computation of cFastLLx and cFastLLy

  using core::models::mhd::FMIN4;
  using core::models::mhd::FMAX4;

  real_t SL = FMIN4(uLL, uLR, uRL, uRR) - FMAX4(cFastLLx, cFastLRx, cFastRLx, cFastRRx);
  real_t SR = FMAX4(uLL, uLR, uRL, uRR) + FMAX4(cFastLLx, cFastLRx, cFastRLx, cFastRRx);
  real_t SB = FMIN4(vLL, vLR, vRL, vRR) - FMAX4(cFastLLy, cFastLRy, cFastRLy, cFastRRy);
  real_t ST = FMAX4(vLL, vLR, vRL, vRR) + FMAX4(cFastLLy, cFastLRy, cFastRLy, cFastRRy);

  /*ELL = uLL*bLL - vLL*aLL;
    ELR = uLR*bLR - vLR*aLR;
    ERL = uRL*bRL - vRL*aRL;
    ERR = uRR*bRR - vRR*aRR;*/

  real_t PtotLL = pLL + HALF_F * (aLL * aLL + bLL * bLL + cLL * cLL);
  real_t PtotLR = pLR + HALF_F * (aLR * aLR + bLR * bLR + cLR * cLR);
  real_t PtotRL = pRL + HALF_F * (aRL * aRL + bRL * bRL + cRL * cRL);
  real_t PtotRR = pRR + HALF_F * (aRR * aRR + bRR * bRR + cRR * cRR);

  real_t rcLLx = rLL * (uLL - SL);
  real_t rcRLx = rRL * (SR - uRL);
  real_t rcLRx = rLR * (uLR - SL);
  real_t rcRRx = rRR * (SR - uRR);
  real_t rcLLy = rLL * (vLL - SB);
  real_t rcLRy = rLR * (ST - vLR);
  real_t rcRLy = rRL * (vRL - SB);
  real_t rcRRy = rRR * (ST - vRR);

  real_t ustar =
    (rcLLx * uLL + rcLRx * uLR + rcRLx * uRL + rcRRx * uRR + (PtotLL - PtotRL + PtotLR - PtotRR)) /
    (rcLLx + rcLRx + rcRLx + rcRRx);
  real_t vstar =
    (rcLLy * vLL + rcLRy * vLR + rcRLy * vRL + rcRRy * vRR + (PtotLL - PtotLR + PtotRL - PtotRR)) /
    (rcLLy + rcLRy + rcRLy + rcRRy);

  real_t rstarLLx = rLL * (SL - uLL) / (SL - ustar);
  real_t BstarLL = bLL * (SL - uLL) / (SL - ustar);
  real_t rstarLLy = rLL * (SB - vLL) / (SB - vstar);
  real_t AstarLL = aLL * (SB - vLL) / (SB - vstar);
  real_t rstarLL = rLL * (SL - uLL) / (SL - ustar) * (SB - vLL) / (SB - vstar);
  real_t EstarLLx = ustar * BstarLL - vLL * aLL;
  real_t EstarLLy = uLL * bLL - vstar * AstarLL;
  real_t EstarLL = ustar * BstarLL - vstar * AstarLL;

  real_t rstarLRx = rLR * (SL - uLR) / (SL - ustar);
  real_t BstarLR = bLR * (SL - uLR) / (SL - ustar);
  real_t rstarLRy = rLR * (ST - vLR) / (ST - vstar);
  real_t AstarLR = aLR * (ST - vLR) / (ST - vstar);
  real_t rstarLR = rLR * (SL - uLR) / (SL - ustar) * (ST - vLR) / (ST - vstar);
  real_t EstarLRx = ustar * BstarLR - vLR * aLR;
  real_t EstarLRy = uLR * bLR - vstar * AstarLR;
  real_t EstarLR = ustar * BstarLR - vstar * AstarLR;

  real_t rstarRLx = rRL * (SR - uRL) / (SR - ustar);
  real_t BstarRL = bRL * (SR - uRL) / (SR - ustar);
  real_t rstarRLy = rRL * (SB - vRL) / (SB - vstar);
  real_t AstarRL = aRL * (SB - vRL) / (SB - vstar);
  real_t rstarRL = rRL * (SR - uRL) / (SR - ustar) * (SB - vRL) / (SB - vstar);
  real_t EstarRLx = ustar * BstarRL - vRL * aRL;
  real_t EstarRLy = uRL * bRL - vstar * AstarRL;
  real_t EstarRL = ustar * BstarRL - vstar * AstarRL;

  real_t rstarRRx = rRR * (SR - uRR) / (SR - ustar);
  real_t BstarRR = bRR * (SR - uRR) / (SR - ustar);
  real_t rstarRRy = rRR * (ST - vRR) / (ST - vstar);
  real_t AstarRR = aRR * (ST - vRR) / (ST - vstar);
  real_t rstarRR = rRR * (SR - uRR) / (SR - ustar) * (ST - vRR) / (ST - vstar);
  real_t EstarRRx = ustar * BstarRR - vRR * aRR;
  real_t EstarRRy = uRR * bRR - vstar * AstarRR;
  real_t EstarRR = ustar * BstarRR - vstar * AstarRR;

  using core::models::mhd::FMAX5;

  real_t calfvenL = FMAX5(fabs(aLR) / sqrt(rstarLRx),
                          fabs(AstarLR) / sqrt(rstarLR),
                          fabs(aLL) / sqrt(rstarLLx),
                          fabs(AstarLL) / sqrt(rstarLL),
                          settings.hydro.smallc);
  real_t calfvenR = FMAX5(fabs(aRR) / sqrt(rstarRRx),
                          fabs(AstarRR) / sqrt(rstarRR),
                          fabs(aRL) / sqrt(rstarRLx),
                          fabs(AstarRL) / sqrt(rstarRL),
                          settings.hydro.smallc);
  real_t calfvenB = FMAX5(fabs(bLL) / sqrt(rstarLLy),
                          fabs(BstarLL) / sqrt(rstarLL),
                          fabs(bRL) / sqrt(rstarRLy),
                          fabs(BstarRL) / sqrt(rstarRL),
                          settings.hydro.smallc);
  real_t calfvenT = FMAX5(fabs(bLR) / sqrt(rstarLRy),
                          fabs(BstarLR) / sqrt(rstarLR),
                          fabs(bRR) / sqrt(rstarRRy),
                          fabs(BstarRR) / sqrt(rstarRR),
                          settings.hydro.smallc);

  real_t SAL = fmin(ustar - calfvenL, ZERO_F);
  real_t SAR = fmax(ustar + calfvenR, ZERO_F);
  real_t SAB = fmin(vstar - calfvenB, ZERO_F);
  real_t SAT = fmax(vstar + calfvenT, ZERO_F);

  real_t AstarT = (SAR * AstarRR - SAL * AstarLR) / (SAR - SAL);
  real_t AstarB = (SAR * AstarRL - SAL * AstarLL) / (SAR - SAL);

  real_t BstarR = (SAT * BstarRR - SAB * BstarRL) / (SAT - SAB);
  real_t BstarL = (SAT * BstarLR - SAB * BstarLL) / (SAT - SAB);

  // finally get emf E
  real_t E = 0, tmpE = 0;

  // the following part is slightly different from the original fortran
  // code since it has to much different branches
  // which generate to much branch divergence in CUDA !!!

  // compute sort of boolean (don't know if signbit is available)
  real_t SB_pos = (1 + COPYSIGN(ONE_F, SB)) / 2, SB_neg = 1 - SB_pos;
  real_t ST_pos = (1 + COPYSIGN(ONE_F, ST)) / 2, ST_neg = 1 - ST_pos;
  real_t SL_pos = (1 + COPYSIGN(ONE_F, SL)) / 2, SL_neg = 1 - SL_pos;
  real_t SR_pos = (1 + COPYSIGN(ONE_F, SR)) / 2, SR_neg = 1 - SR_pos;

  // else
  tmpE = (SAL * SAB * EstarRR - SAL * SAT * EstarRL - SAR * SAB * EstarLR + SAR * SAT * EstarLL) /
           (SAR - SAL) / (SAT - SAB) -
         SAT * SAB / (SAT - SAB) * (AstarT - AstarB) + SAR * SAL / (SAR - SAL) * (BstarR - BstarL);
  E += (SB_neg * ST_pos * SL_neg * SR_pos) * tmpE;

  // SB>0
  tmpE = (SAR * EstarLLx - SAL * EstarRLx + SAR * SAL * (bRL - bLL)) / (SAR - SAL);
  tmpE = SL_pos * ELL + SL_neg * SR_neg * ERL + SL_neg * SR_pos * tmpE;
  E += SB_pos * tmpE;

  // ST<0
  tmpE = (SAR * EstarLRx - SAL * EstarRRx + SAR * SAL * (bRR - bLR)) / (SAR - SAL);
  tmpE = SL_pos * ELR + SL_neg * SR_neg * ERR + SL_neg * SR_pos * tmpE;
  E += (SB_neg * ST_neg) * tmpE;

  // SL>0
  tmpE = (SAT * EstarLLy - SAB * EstarLRy - SAT * SAB * (aLR - aLL)) / (SAT - SAB);
  E += (SB_neg * ST_pos * SL_pos) * tmpE;

  // SR<0
  tmpE = (SAT * EstarRLy - SAB * EstarRRy - SAT * SAB * (aRR - aRL)) / (SAT - SAB);
  E += (SB_neg * ST_pos * SL_neg * SR_neg) * tmpE;


  /*
    if(SB>ZERO_F) {
    if(SL>ZERO_F) {
    E=ELL;
    } else if(SR<ZERO_F) {
    E=ERL;
    } else {
    E=(SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
    }
    } else if (ST<ZERO_F) {
    if(SL>ZERO_F) {
    E=ELR;
    } else if(SR<ZERO_F) {
    E=ERR;
    } else {
    E=(SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
    }
    } else if (SL>ZERO_F) {
    E=(SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
    } else if (SR<ZERO_F) {
    E=(SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
    } else {
    E = (SAL*SAB*EstarRR-SAL*SAT*EstarRL -
    SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) -
    SAT*SAB/(SAT-SAB)*(AstarT-AstarB) +
    SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
    }
  */

  return E;

} // mag_riemann2d_hlld

/**
 * Compute emf from qEdge state vector via a 2D magnetic Riemann
 * solver (see routine cmp_mag_flux in DUMSES).
 *
 * \param[in] qEdge array containing input states qRT, qLT, qRB, qLB
 * \param[in] xPos x position in space (only needed for shearing box correction terms).
 * \return emf
 *
 * template parameters:
 *
 * \tparam emfDir plays the role of xdim/lor in DUMSES routine
 * cmp_mag_flx, i.e. define which EMF will be computed (how to define
 * parallel/orthogonal velocity). emfDir identifies the orthogonal direction.
 *
 * \note the global parameter magRiemannSolver is used to choose the
 * 2D magnetic Riemann solver.
 *
 * TODO: make xPos parameter non-optional
 */
template <EmfDir emfDir>
KOKKOS_INLINE_FUNCTION real_t
compute_emf(MHDStateCell (&qEdge)[4],
            MHDSettings const &     settings,
            [[maybe_unused]] real_t xPos = 0)
{

  // makes enum Hydro::VarId available
  using MHD = kalypsso::core::models::MHD;

  // define alias reference to input arrays
  MHDStateCell & qRT = qEdge[IRT];
  MHDStateCell & qLT = qEdge[ILT];
  MHDStateCell & qRB = qEdge[IRB];
  MHDStateCell & qLB = qEdge[ILB];

  // defines alias reference to intermediate state before applying a
  // magnetic Riemann solver
  MHDStateCell   qLLRR[4];
  MHDStateCell & qLL = qLLRR[ILL];
  MHDStateCell & qRL = qLLRR[IRL];
  MHDStateCell & qLR = qLLRR[ILR];
  MHDStateCell & qRR = qLLRR[IRR];

  // density
  qLL[MHD::ID] = qRT[MHD::ID];
  qRL[MHD::ID] = qLT[MHD::ID];
  qLR[MHD::ID] = qRB[MHD::ID];
  qRR[MHD::ID] = qLB[MHD::ID];

  // pressure
  // ISOTHERMAL
  real_t cIso = settings.hydro.cIso;
  if (cIso > 0)
  {
    qLL[MHD::IP] = qLL[MHD::ID] * cIso * cIso;
    qRL[MHD::IP] = qRL[MHD::ID] * cIso * cIso;
    qLR[MHD::IP] = qLR[MHD::ID] * cIso * cIso;
    qRR[MHD::IP] = qRR[MHD::ID] * cIso * cIso;
  }
  else
  {
    qLL[MHD::IP] = qRT[MHD::IP];
    qRL[MHD::IP] = qLT[MHD::IP];
    qLR[MHD::IP] = qRB[MHD::IP];
    qRR[MHD::IP] = qLB[MHD::IP];
  }

  // iu, iv : parallel velocity indexes
  // iw     : orthogonal velocity index
  // ia, ib, ic : idem for magnetic field
  // int iu, iv, iw, ia, ib, ic;
  if (emfDir == EMFZ)
  {

    // iu = MHD::IU; iv = MHD::IV; iw = MHD::IW;
    // ia = MHD::IA; ib = MHD::IB, ic = MHD::IC;

    // First parallel velocity
    qLL[MHD::IU] = qRT[MHD::IU];
    qRL[MHD::IU] = qLT[MHD::IU];
    qLR[MHD::IU] = qRB[MHD::IU];
    qRR[MHD::IU] = qLB[MHD::IU];

    // Second parallel velocity
    qLL[MHD::IV] = qRT[MHD::IV];
    qRL[MHD::IV] = qLT[MHD::IV];
    qLR[MHD::IV] = qRB[MHD::IV];
    qRR[MHD::IV] = qLB[MHD::IV];

    // First parallel magnetic field (enforce continuity)
    qLL[MHD::IA] = HALF_F * (qRT[MHD::IA] + qLT[MHD::IA]);
    qRL[MHD::IA] = HALF_F * (qRT[MHD::IA] + qLT[MHD::IA]);
    qLR[MHD::IA] = HALF_F * (qRB[MHD::IA] + qLB[MHD::IA]);
    qRR[MHD::IA] = HALF_F * (qRB[MHD::IA] + qLB[MHD::IA]);

    // Second parallel magnetic field (enforce continuity)
    qLL[MHD::IB] = HALF_F * (qRT[MHD::IB] + qRB[MHD::IB]);
    qRL[MHD::IB] = HALF_F * (qLT[MHD::IB] + qLB[MHD::IB]);
    qLR[MHD::IB] = HALF_F * (qRT[MHD::IB] + qRB[MHD::IB]);
    qRR[MHD::IB] = HALF_F * (qLT[MHD::IB] + qLB[MHD::IB]);

    // Orthogonal velocity
    qLL[MHD::IW] = qRT[MHD::IW];
    qRL[MHD::IW] = qLT[MHD::IW];
    qLR[MHD::IW] = qRB[MHD::IW];
    qRR[MHD::IW] = qLB[MHD::IW];

    // Orthogonal magnetic Field
    qLL[MHD::IC] = qRT[MHD::IC];
    qRL[MHD::IC] = qLT[MHD::IC];
    qLR[MHD::IC] = qRB[MHD::IC];
    qRR[MHD::IC] = qLB[MHD::IC];
  }
  else if (emfDir == EMFY)
  {

    // iu = MHD::IW; iv = MHD::IU; iw = MHD::IV;
    // ia = MHD::IC; ib = MHD::IA, ic = MHD::IB;

    // First parallel velocity
    qLL[MHD::IU] = qRT[MHD::IW];
    qRL[MHD::IU] = qLT[MHD::IW];
    qLR[MHD::IU] = qRB[MHD::IW];
    qRR[MHD::IU] = qLB[MHD::IW];

    // Second parallel velocity
    qLL[MHD::IV] = qRT[MHD::IU];
    qRL[MHD::IV] = qLT[MHD::IU];
    qLR[MHD::IV] = qRB[MHD::IU];
    qRR[MHD::IV] = qLB[MHD::IU];

    // First parallel magnetic field (enforce continuity)
    qLL[MHD::IA] = HALF_F * (qRT[MHD::IC] + qLT[MHD::IC]);
    qRL[MHD::IA] = HALF_F * (qRT[MHD::IC] + qLT[MHD::IC]);
    qLR[MHD::IA] = HALF_F * (qRB[MHD::IC] + qLB[MHD::IC]);
    qRR[MHD::IA] = HALF_F * (qRB[MHD::IC] + qLB[MHD::IC]);

    // Second parallel magnetic field (enforce continuity)
    qLL[MHD::IB] = HALF_F * (qRT[MHD::IA] + qRB[MHD::IA]);
    qRL[MHD::IB] = HALF_F * (qLT[MHD::IA] + qLB[MHD::IA]);
    qLR[MHD::IB] = HALF_F * (qRT[MHD::IA] + qRB[MHD::IA]);
    qRR[MHD::IB] = HALF_F * (qLT[MHD::IA] + qLB[MHD::IA]);

    // Orthogonal velocity
    qLL[MHD::IW] = qRT[MHD::IV];
    qRL[MHD::IW] = qLT[MHD::IV];
    qLR[MHD::IW] = qRB[MHD::IV];
    qRR[MHD::IW] = qLB[MHD::IV];

    // Orthogonal magnetic Field
    qLL[MHD::IC] = qRT[MHD::IB];
    qRL[MHD::IC] = qLT[MHD::IB];
    qLR[MHD::IC] = qRB[MHD::IB];
    qRR[MHD::IC] = qLB[MHD::IB];
  }
  else
  { // emfDir == EMFX

    // iu = MHD::IV; iv = MHD::IW; iw = MHD::IU;
    // ia = MHD::IB; ib = MHD::IC, ic = MHD::IA;

    // First parallel velocity
    qLL[MHD::IU] = qRT[MHD::IV];
    qRL[MHD::IU] = qLT[MHD::IV];
    qLR[MHD::IU] = qRB[MHD::IV];
    qRR[MHD::IU] = qLB[MHD::IV];

    // Second parallel velocity
    qLL[MHD::IV] = qRT[MHD::IW];
    qRL[MHD::IV] = qLT[MHD::IW];
    qLR[MHD::IV] = qRB[MHD::IW];
    qRR[MHD::IV] = qLB[MHD::IW];

    // First parallel magnetic field (enforce continuity)
    qLL[MHD::IA] = HALF_F * (qRT[MHD::IB] + qLT[MHD::IB]);
    qRL[MHD::IA] = HALF_F * (qRT[MHD::IB] + qLT[MHD::IB]);
    qLR[MHD::IA] = HALF_F * (qRB[MHD::IB] + qLB[MHD::IB]);
    qRR[MHD::IA] = HALF_F * (qRB[MHD::IB] + qLB[MHD::IB]);

    // Second parallel magnetic field (enforce continuity)
    qLL[MHD::IB] = HALF_F * (qRT[MHD::IC] + qRB[MHD::IC]);
    qRL[MHD::IB] = HALF_F * (qLT[MHD::IC] + qLB[MHD::IC]);
    qLR[MHD::IB] = HALF_F * (qRT[MHD::IC] + qRB[MHD::IC]);
    qRR[MHD::IB] = HALF_F * (qLT[MHD::IC] + qLB[MHD::IC]);

    // Orthogonal velocity
    qLL[MHD::IW] = qRT[MHD::IU];
    qRL[MHD::IW] = qLT[MHD::IU];
    qLR[MHD::IW] = qRB[MHD::IU];
    qRR[MHD::IW] = qLB[MHD::IU];

    // Orthogonal magnetic Field
    qLL[MHD::IC] = qRT[MHD::IA];
    qRL[MHD::IC] = qLT[MHD::IA];
    qLR[MHD::IC] = qRB[MHD::IA];
    qRR[MHD::IC] = qLB[MHD::IA];
  }

  // Compute final fluxes

  // vx*by - vy*bx at the four edge centers
  real_t   eLLRR[4];
  real_t & ELL = eLLRR[ILL];
  real_t & ERL = eLLRR[IRL];
  real_t & ELR = eLLRR[ILR];
  real_t & ERR = eLLRR[IRR];

  ELL = qLL[MHD::IU] * qLL[MHD::IB] - qLL[MHD::IV] * qLL[MHD::IA];
  ERL = qRL[MHD::IU] * qRL[MHD::IB] - qRL[MHD::IV] * qRL[MHD::IA];
  ELR = qLR[MHD::IU] * qLR[MHD::IB] - qLR[MHD::IV] * qLR[MHD::IA];
  ERR = qRR[MHD::IU] * qRR[MHD::IB] - qRR[MHD::IV] * qRR[MHD::IA];

  real_t emf = 0;
  // mag_riemann2d<>
  // if (params.magRiemannSolver == MAG_HLLD) {
  emf = mag_riemann2d_hlld(qLLRR, eLLRR, settings);
  // } else if (params.magRiemannSolver == MAG_HLLA) {
  //   emf = mag_riemann2d_hlla(qLLRR, eLLRR);
  // } else if (params.magRiemannSolver == MAG_HLLF) {
  //   emf = mag_riemann2d_hllf(qLLRR, eLLRR);
  // } else if (params.magRiemannSolver == MAG_LLF) {
  //   emf = mag_riemann2d_llf(qLLRR, eLLRR);
  // }

  return emf;

} // compute_emf

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_RIEMANN_SOLVERS_MHD_H_
