// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BlastParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_BLAST_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_BLAST_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>

namespace kalypsso
{

/**
 * (Sedov) Blast test parameters.
 *
 * We provide two types of initializations:
 *
 * - if total_energy_inside is greater than 0, then we initialize pressure by
 * (gamma-1)*e/volume_inside where volume_inside is total volume of the inside ball
 * - if total_energy_inside is zero, then we simply used blast_pressure_in to initialize pressure
 *
 * When performing a quantitative Sedov test, you need to use the first initialization to properly
 * control the energy deposit inside the ball, and chose the ball radius as small as possible (e.g.
 * 1/100th of the box size).
 *
 * \note this test can also be run in MHD using a uniform magnetic field.
 * Reference:
 * "The Athena++ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers", James
 * M. Stone et al 2020 ApJS 249 4,  https://iopscience.iop.org/article/10.3847/1538-4365/ab929b
 * section 3.4.4
 * DOI 10.3847/1538-4365/ab929b
 *
 */
struct BlastParams
{

  // blast problem parameters
  real_t blast_radius;
  real_t blast_center_x;
  real_t blast_center_y;
  real_t blast_center_z;
  real_t blast_density_in;
  real_t blast_density_out;
  real_t blast_pressure_in;
  real_t blast_pressure_out;
  real_t total_energy_inside = ZERO_F;
  real_t bx;
  real_t by;
  real_t bz;

  BlastParams(ConfigMap const & config_map)
  {

    const auto xmin = config_map.getReal("mesh", "xmin", ZERO_F);
    const auto ymin = config_map.getReal("mesh", "ymin", ZERO_F);
    const auto zmin = config_map.getReal("mesh", "zmin", ZERO_F);

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    blast_radius = config_map.getReal("blast", "radius", (xmin + xmax) / TWO_F / 10);
    blast_center_x = config_map.getReal("blast", "center_x", (xmin + xmax) / 2);
    blast_center_y = config_map.getReal("blast", "center_y", (ymin + ymax) / 2);
    blast_center_z = config_map.getReal("blast", "center_z", (zmin + zmax) / 2);

    blast_density_in = config_map.getReal("blast", "density_in", KALYPSSO_NUM(1.0));
    blast_density_out = config_map.getReal("blast", "density_out", KALYPSSO_NUM(1.2));
    blast_pressure_in = config_map.getReal("blast", "pressure_in", KALYPSSO_NUM(10.0));
    blast_pressure_out = config_map.getReal("blast", "pressure_out", KALYPSSO_NUM(0.1));
    total_energy_inside = config_map.getReal("blast", "total_energy_inside", ZERO_F);
    bx = config_map.getReal("blast", "bx", ONE_F);
    by = config_map.getReal("blast", "by", SQRT_3_F);
    bz = config_map.getReal("blast", "bz", ZERO_F);
  } // BlastParams

}; // struct BlastParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_BLAST_PARAMS_H_
