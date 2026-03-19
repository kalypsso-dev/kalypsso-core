// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file KHParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_KELVIN_HELMHOLTZ_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_KELVIN_HELMHOLTZ_PARAMS_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_MPI
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * A small structure to hold parameters passed to a Kokkos functor,
 * for initializing the Kelvin-Helmholtz instability init condition.
 *
 * p_sine, p_sine_robertson and p_rand specify which type of perturbation is
 * used to seed the instability.
 *
 * references:
 * - "Computational Eulerian hydrodynamics and Galilean invariance", Robertson et al, MNRAS Volume
 * 401, Issue 4, February 2010; https://doi.org/10.1111/j.1365-2966.2009.15823.x
 * - "The Athena++ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers", J.
 * stone et al, 2020 ApJS 249 4; https://doi.org/10.3847/1538-4365/ab929b
 */
struct KHParams
{

  // Kelvin-Helmholtz problem parameters
  real_t d_in;  //! density in
  real_t d_out; //! density out
  real_t pressure;
  bool   p_sine;       //! sinus perturbation
  bool   p_sine_rob;   //! sinus perturbation "à la Robertson"
  bool   p_sine_stone; //! sinus perturbation "à la Stone 2020"
  bool   p_rand;       //! random perturbation

  real_t vflow_in;
  real_t vflow_out;

  uint64_t seed;
  real_t   amplitude; //! perturbation amplitude
  real_t   outer_size;
  real_t   inner_size;

  // for sine perturbation "a la Robertson"
  int    mode;
  real_t w0;
  real_t delta;

  real_t bx;
  real_t by;
  real_t bz;

  KHParams(ConfigMap const & config_map)
  {

    d_in = config_map.getReal("kelvin_helmholtz", "d_in", KALYPSSO_NUM(1.0));
    d_out = config_map.getReal("kelvin_helmholtz", "d_out", KALYPSSO_NUM(2.0));

    pressure = config_map.getReal("kelvin_helmholtz", "pressure", KALYPSSO_NUM(10.0));

    p_sine = config_map.getBool("kelvin_helmholtz", "perturbation_sine", false);
    p_sine_rob = config_map.getBool("kelvin_helmholtz", "perturbation_sine_robertson", true);
    p_sine_stone = config_map.getBool("kelvin_helmholtz", "perturbation_sine_stone", false);
    p_rand = config_map.getBool("kelvin_helmholtz", "perturbation_rand", false);


    vflow_in = config_map.getReal("kelvin_helmholtz", "vflow_in", KALYPSSO_NUM(-0.5));
    vflow_out = config_map.getReal("kelvin_helmholtz", "vflow_out", KALYPSSO_NUM(0.5));

    if (p_rand)
    {
      // choose a different random seed per mpi rank
      seed = static_cast<uint64_t>(config_map.getInteger("kelvin_helmholtz", "rand_seed", 12));

#ifdef KALYPSSO_CORE_USE_MPI
      // srand( seed * (mpiRank+1) );

      // get MPI rank in MPI_COMM_WORLD
      // TODO : pass communicator to the constructor (?)
      int mpiRank = 1;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
      seed *= static_cast<uint64_t>(mpiRank + 1);
#endif // KALYPSSO_CORE_USE_MPI
    }

    amplitude = config_map.getReal("kelvin_helmholtz", "amplitude", KALYPSSO_NUM(0.1));

    if (p_sine_rob or p_sine)
    {
      // perturbation mode number
      inner_size = config_map.getReal("kelvin_helmholtz", "inner_size", KALYPSSO_NUM(0.2));

      mode = config_map.getInteger("kelvin_helmholtz", "mode", 2);
      w0 = config_map.getReal("kelvin_helmholtz", "w0", KALYPSSO_NUM(0.1));
      delta = config_map.getReal("kelvin_helmholtz", "delta", KALYPSSO_NUM(0.03));
    }

    bx = config_map.getReal("kelvin_helmholtz", "bx", KALYPSSO_NUM(0.0));
    by = config_map.getReal("kelvin_helmholtz", "by", KALYPSSO_NUM(0.0));
    bz = config_map.getReal("kelvin_helmholtz", "bz", KALYPSSO_NUM(0.0));

  } // KHParams

}; // struct KHParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_KELVIN_HELMHOLTZ_PARAMS_H_
