// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroParams.h
 * \brief Hydrodynamics solver parameters.
 */
#ifndef KALYPSSO_CORE_HYDRO_PARAMS_H_
#define KALYPSSO_CORE_HYDRO_PARAMS_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/models/riemann_solver_types.h>


#include <stdbool.h>

namespace kalypsso
{

/**
 * Read dimension from parameter file.
 */
size_t
get_dim(ConfigMap const & config_map);

// ===========================================================================
// ===========================================================================
/**
 * Hydro Parameters (declaration).
 */
struct HydroParams
{
  // run parameters

  //! maximum number of time steps.
  int nStepmax;

  //! final simulation time.
  real_t tEnd;

  //! number of outputs.
  //! - nOutput<0  : output at every time step iteration
  //! - nOutput=0  : no output at all
  //! - nOutput>0  : one output every Delta_t = (t_end - t_beg)/nOutput
  int nOutput;

  //! enable output file write.
  bool enableOutput;

  int nlog; /*!<  number of time step iterations between 2 consecutive logs. */

  // int nbvar; /*!< number of conservative variables. */

  size_t dimType; //!< 2D or 3D.

  real_t xmin; /*!< domain bound */
  real_t xmax; /*!< domain bound */
  real_t ymin; /*!< domain bound */
  real_t ymax; /*!< domain bound */
  real_t zmin; /*!< domain bound */
  real_t zmax; /*!< domain bound */

  // AMR related parameter
  int level_min;
  int level_max;

  // switch on/off AMR cycle
  bool amr_cycle_enabled;

  // IO parameters
  bool output_hdf5_enabled;     /*!< enable HDF5 output file format.*/
  bool output_exabrick_enabled; /*!< enable exabrick output file format.*/
  bool debug_output;            /*!< more verbose output */

  // Update type for the hydro solver (conservative / non-conservative)
  int updateType;

  bool replicated_init_cond; /*!< if true, init cond is replicated identically in all trees  (useful
                               for weak scaling studies) */

  HydroParams(ConfigMap const & config_map);

  ~HydroParams() = default;

  //! This is the genuine initialization / setup (fed by parameter file)
  void
  setup(ConfigMap const & config_map);

  void
  print() const;

}; // struct HydroParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_HYDRO_PARAMS_H_
