// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RayleighTaylorParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_RAYLEIGH_TAYLOR_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_RAYLEIGH_TAYLOR_PARAMS_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_MPI
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <../../better-enums/enum.h>

#include <algorithm> // for std::transform
#include <string>    // for ::toupper

namespace kalypsso
{

// clang-format off
BETTER_ENUM(RayleighTaylorPerturbationType, int, UNDEFINED, SINE, RANDOM)
// clang-format on

//! Rayleigh-Taylor problem parameters.
//!
//! There are 2 possibles initializations:
//! - regular    initialization: two fluids one above the other, each with uniform density
//! - isothermal initialization: two fluids one above the other, each with uniform temperature
//!
//! For each of these initialization, the initial transverse velocity is perturbed either using a
//! sine mode or random values.
struct RayleighTaylorParams
{
  //! get initital perturbation type (SINE or RANDOM)
  static inline RayleighTaylorPerturbationType
  get_perturbation_type(ConfigMap const & config_map)
  {
    auto perturbationTypeStr = config_map.getString("rayleigh_taylor", "perturbation", "SINE");
    std::transform(perturbationTypeStr.begin(),
                   perturbationTypeStr.end(),
                   perturbationTypeStr.begin(),
                   ::toupper);

    auto maybe_value =
      RayleighTaylorPerturbationType::_from_string_nothrow(perturbationTypeStr.c_str());
    if (maybe_value)
    {
      return *maybe_value;
    }
    else
    {
      return RayleighTaylorPerturbationType::SINE;
    }
  }

  real_t amplitude; //!< perturbation amplitude
  int    nx;        //!< mode number along x, use nx > 1 for a multi mode perturbation
  int    ny;        //!< mode number along y, use ny > 1 for a multi mode perturbation
  int    nz;        //!< mode number along z, use nz > 1 for a multi mode perturbation (3D only)
  real_t rho_up;    //!< fluid density above interface
  real_t rho_down;  //!< fluid density below interface
  real_t P0;        //!< reference pressure
  RayleighTaylorPerturbationType perturb_type; //!< perturbation type (sine or random)
  uint64_t                       seed;         //!< random seed
  real_t                         bx;           //!< magnetic field along x
  real_t                         by;           //!< magnetic field along y
  real_t                         bz;           //!< magnetic field along z

  bool   use_isothermal_init; //!< use isothermal equilibrium or not (iso-density)
  real_t T_up;   //!< fluid temperature above interface (only valid with isothermal init)
  real_t T_down; //!< fluid temperature below interface (only valid with isothermal init)


  RayleighTaylorParams(ConfigMap const & config_map)
    : amplitude(config_map.getReal("rayleigh_taylor", "amplitude", KALYPSSO_NUM(0.01)))
    , nx(config_map.getInteger("rayleigh_taylor", "nx", 1))
    , ny(config_map.getInteger("rayleigh_taylor", "ny", 1))
    , nz(config_map.getInteger("rayleigh_taylor", "nz", 1))
    , rho_up(config_map.getReal("rayleigh_taylor", "rho_up", KALYPSSO_NUM(2.0)))
    , rho_down(config_map.getReal("rayleigh_taylor", "rho_down", KALYPSSO_NUM(1.0)))
    , P0(config_map.getReal("rayleigh_taylor", "P0", KALYPSSO_NUM(2.5)))
    , perturb_type(get_perturbation_type(config_map))
    , seed(static_cast<uint64_t>(config_map.getInteger("rayleigh_taylor", "rand_seed", 12)))
    , bx(config_map.getReal("rayleigh_taylor", "bx", KALYPSSO_NUM(0.0)))
    , by(config_map.getReal("rayleigh_taylor", "by", KALYPSSO_NUM(0.0)))
    , bz(config_map.getReal("rayleigh_taylor", "bz", KALYPSSO_NUM(0.0)))
    , use_isothermal_init(config_map.getBool("rayleigh_taylor", "use_isothermal_init", false))
    , T_up(config_map.getReal("rayleigh_taylor", "T_up", KALYPSSO_NUM(1.0)))
    , T_down(config_map.getReal("rayleigh_taylor", "T_down", KALYPSSO_NUM(2.0)))
  {

    // choose a different random seed per mpi rank
    seed = static_cast<uint64_t>(config_map.getInteger("rayleigh_taylor", "rand_seed", 12));

#ifdef KALYPSSO_CORE_USE_MPI
    // srand( seed * (mpiRank+1) );

    // get MPI rank in MPI_COMM_WORLD
    // TODO : pass communicator to the constructor (?)
    int mpiRank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    seed *= static_cast<uint64_t>(mpiRank + 1);
#endif // KALYPSSO_CORE_USE_MPI
  }

}; // struct RayleighTaylorParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_RAYLEIGH_TAYLOR_PARAMS_H_
