// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroSettings.h
 * \brief Hydrodynamics solver parameters.
 */
#ifndef KALYPSSO_CORE_MODELS_MHDSETTINGS_H_
#define KALYPSSO_CORE_MODELS_MHDSETTINGS_H_

#include <kalypsso/core/models/HydroSettings.h>

namespace kalypsso
{

// ===========================================================================
// ===========================================================================
/**
 * Parameters that can be passed by copy to a Kokkos device.
 */
struct MHDSettings
{
  //! Hydrodynamics settings
  HydroSettings hydro;

  //! enable the splitting of magnetic field: B = B0 + B1
  bool mag_field_split_enabled;

  //! enable Boris correction
  bool correction_boris_enabled;

  //! velocity used in the boris correction
  real_t cboris;

  //! enable entropy correction
  bool correction_entropy_enabled;

  //! Lower threshold value for plasma beta
  real_t small_beta;

  //! Upper threshold value for alfven
  real_t large_alfven;

  MHDSettings(ConfigMap const & config_map);
  ~MHDSettings() = default;

  void
  print() const;

}; // struct MHDSettings

} // namespace kalypsso

#endif // KALYPSSO_CORE_MODELS_MHDSETTINGS_H_
