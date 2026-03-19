// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file initRiemannConfig2d.h
 * \brief Implement initialization routine to solve a four quadrant 2D Riemann
 * problem.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 */
#ifndef KALYPSSO_CORE_PROBLEMS_INIT_RIEMANN_CONFIG_2D_H_
#define KALYPSSO_CORE_PROBLEMS_INIT_RIEMANN_CONFIG_2D_H_

#include <kalypsso/core/real_type.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroState.h>

namespace kalypsso
{

// =====================================================================
// =====================================================================
template <size_t dim>
constexpr uint32_t
nb_riemann_states()
{
  if constexpr (dim == 2)
  {
    return 4;
  }
  else if constexpr (dim == 3)
  {
    return 8;
  }
} // nb_riemann_states

// =====================================================================
// =====================================================================
// =====================================================================
template <size_t dim>
struct RiemannConfig
{
  static constexpr uint32_t NB_STATES = nb_riemann_states<dim>();

  using HydroState_t = HydroState<dim>;

  using HydroStates_t = Kokkos::Array<HydroState_t, nb_riemann_states<dim>()>;
};

// =====================================================================
// =====================================================================
/**
 * [DEPRECATED] Convert primitive to conservative variables.
 *
 * Each numerical scheme should provide this routine using an EOS wrapper for computing internal
 * energy from pressure.
 */
template <size_t dim>
KOKKOS_FUNCTION void
primToCons(HydroState<dim> & U, real_t gamma0)
{

  constexpr auto ID = core::models::Hydro::ID;
  constexpr auto IP = core::models::Hydro::IP;
  constexpr auto IU = core::models::Hydro::IU;
  constexpr auto IV = core::models::Hydro::IV;
  constexpr auto IW = core::models::Hydro::IW;

  real_t rho = U[ID];
  real_t p = U[IP];
  real_t u = U[IU];
  real_t v = U[IV];

  if constexpr (dim == 2)
  {
    U[IU] *= rho; // rho*u
    U[IV] *= rho; // rho*v
    U[IP] = p / (gamma0 - ONE_F) + rho * (u * u + v * v) * HALF_F;
  }
  else if constexpr (dim == 3)
  {
    real_t w = U[IW];
    U[IU] *= rho; // rho*u
    U[IV] *= rho; // rho*v
    U[IW] *= rho; // rho*w
    U[IP] = p / (gamma0 - ONE_F) + rho * (u * u + v * v + w * w) * HALF_F;
  }

} // primToCons

// =====================================================================
// =====================================================================
/**
 * Return an array of HydroStates, one per region, in primitive variables
 */
KOKKOS_FUNCTION
RiemannConfig<2>::HydroStates_t
getRiemannConfig2d(int numConfig);

// =====================================================================
// =====================================================================
/**
 * Return an array of HydroStates, one per region, in primitive variables
 */
KOKKOS_FUNCTION
RiemannConfig<3>::HydroStates_t
getRiemannConfig3d(int numConfig);

// =====================================================================
// =====================================================================
/**
 * Return an array of HydroStates, one per region, in primitive variables
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION auto
getRiemannConfig(int numConfig)
{

  if constexpr (dim == 2)
  {
    return getRiemannConfig2d(numConfig);
  }
  else if constexpr (dim == 3)
  {
    return getRiemannConfig3d(numConfig);
  }
} // getRiemannConfig

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_INIT_RIEMANN_CONFIG_2D_H_
