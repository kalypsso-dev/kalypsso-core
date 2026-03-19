// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DiamagCavityParams.h
 */
#ifndef KALYPSSO_CORE_PROBLEMS_DIAMAG_CAVITY_PARAMS_H_
#define KALYPSSO_CORE_PROBLEMS_DIAMAG_CAVITY_PARAMS_H_

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/physical_constants.h>

#include <algorithm>
#include <../../better-enums/enum.h>

namespace kalypsso
{

/**
 * Cavity shape.
 */
// clang-format off
BETTER_ENUM(DiamagCavityShape, int, SPHERE, CYLINDER)
// clang-format on

/**
 * Density profile type.
 */
// clang-format off
BETTER_ENUM(RhoProfile, int, BASE = 1, MODIFIED = 2)
// clang-format on


/**
 * Provide all parameters to init the diamagnetic cavity test case.
 */
struct DiamagCavityParams
{

  //! get shape
  static inline DiamagCavityShape
  get_shape(ConfigMap const & config_map)
  {
    auto shapeStr = config_map.getString("cavity", "shape", "SPHERE");
    std::transform(shapeStr.begin(), shapeStr.end(), shapeStr.begin(), ::toupper);

    auto maybe_value = DiamagCavityShape::_from_string_nothrow(shapeStr.c_str());
    if (maybe_value)
    {
      return *maybe_value;
    }

    // default
    return DiamagCavityShape::SPHERE;
  }

  //! get shape
  static inline RhoProfile
  get_rho_profile(ConfigMap const & config_map)
  {
    auto rhoProfileStr = config_map.getString("cavity", "rho_profile", "BASE");
    std::transform(rhoProfileStr.begin(), rhoProfileStr.end(), rhoProfileStr.begin(), ::toupper);

    auto maybe_value = RhoProfile::_from_string_nothrow(rhoProfileStr.c_str());
    if (maybe_value)
    {
      return *maybe_value;
    }

    // default
    return RhoProfile::BASE;
  }

  // Diamagnetic cavity parameters
  real_t center_x; //!< Center of the cavity in the x axis
  real_t center_y; //!< Center of the cavity in the y axis
  real_t center_z; //!< Center of the cavity in the z axis

  real_t t0; //!< Physical starting time [s]

  real_t Vmin; //!< Minimal Value of the initial debris velocity [m/s]
  real_t Vmax; //!< Maximal value of the initial debris Velocity [m/s]
  real_t dV;   //!< Velocity variation in the initial debris cloud [s^(-1)]

  real_t rmin; //!< Minimal radius of the initial spread of the debris [m]
  real_t rmax; //!< Minimal radius of the initial spread of the debris [m]

  real_t M_d; //!< Debris molar mass  (default is Carbon) [kg/mol]
  real_t m_d; //!< Debris mass of one atom/ion [kg]
  real_t N_d; //!< Debris ion number [#]
  real_t T_d; //!< Debris initial temperature equivalent to Ti 1eV=11300K [K]

  real_t M_a;   //!< Ambient molar mass (default is Hydrogen) [kg/mol]
  real_t m_a;   //!< Ambient mass of one atom/ion [kg]
  real_t n_a;   //!< Ambient ion density [#/m^(-3)]
  real_t N_a;   //!< Ambient ion number [#]
  real_t rho_a; //!< Ambient ion mass density [kg/m^3]

  real_t B0; //!< Ambient amplitude of magnetic field [T]
  real_t Bx;
  real_t By;
  real_t Bz;

  real_t amplitude_perturb; //! Amplitude of the perturbation of initial density
  real_t wave_number;       //! wave number of the perturbation of initial density
  bool   perturb;           //! add a sinusoidinis perturbation on initial density contour

  RhoProfile rho_profile;

  DiamagCavityShape shape;

  DiamagCavityParams(ConfigMap const & config_map)
    : center_x(ZERO_F)
    , center_y(ZERO_F)
    , center_z(ZERO_F)
    , t0(config_map.getReal("cavity", "t0", KALYPSSO_NUM(2e-7)))
    , Vmin(config_map.getReal("cavity", "Vmin", ZERO_F))
    , Vmax(config_map.getReal("cavity", "Vmax", KALYPSSO_NUM(260e3)))
    , dV(ZERO_F)
    , rmin(config_map.getReal("cavity", "rmin", ZERO_F))
    , rmax(config_map.getReal("cavity", "rmax", Vmax * t0))
    , M_d(config_map.getReal("cavity", "M_d", KALYPSSO_NUM(0.012011)))
    , m_d(ZERO_F)
    , N_d(config_map.getReal("cavity", "N_d", KALYPSSO_NUM(1.5608816704478538e+18)))
    , T_d(config_map.getReal("cavity", "T_d", KALYPSSO_NUM(11604.0)))
    , M_a(config_map.getReal("cavity", "M_a", KALYPSSO_NUM(1.008e-3)))
    , m_a(ZERO_F)
    , n_a(config_map.getReal("cavity", "n_a", KALYPSSO_NUM(5e18)))
    , N_a(ZERO_F)
    , rho_a(ZERO_F)
    , B0(config_map.getReal("cavity", "B0", KALYPSSO_NUM(0.02)))
    , Bx(config_map.getReal("cavity", "Bx", ZERO_F))
    , By(config_map.getReal("cavity", "By", ZERO_F))
    , Bz(config_map.getReal("cavity", "Bz", B0 / sqrt(constants::MU0)))
    , amplitude_perturb(config_map.getReal("cavity", "amplitude_perturb", KALYPSSO_NUM(0.0025)))
    , wave_number(config_map.getReal("cavity", "wave_number", KALYPSSO_NUM(10.0)))
    , perturb(config_map.getBool("cavity", "perturb", false))
    , rho_profile(get_rho_profile(config_map))
    , shape(get_shape(config_map))
  {
    const auto xmin = config_map.getReal("mesh", "xmin", ZERO_F);
    const auto ymin = config_map.getReal("mesh", "ymin", ZERO_F);
    const auto zmin = config_map.getReal("mesh", "zmin", ZERO_F);

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", 1.0);

    const auto xmax = xmin + nbrick_x * scaling_factor;
    const auto ymax = ymin + nbrick_y * scaling_factor;
    const auto zmax = zmin + nbrick_z * scaling_factor;

    center_x = config_map.getReal("cavity", "center_x", (xmin + xmax) / 2);
    center_y = config_map.getReal("cavity", "center_y", (ymin + ymax) / 2);
    center_z = config_map.getReal("cavity", "center_z", (zmin + zmax) / 2);

    // Values deduced from inputs
    dV = (Vmax - Vmin) / (rmax - rmin);

    m_d = M_d / constants::AVOGADRO;

    m_a = M_a / constants::AVOGADRO;
    N_a = n_a * PI_F * rmax * rmax;
    rho_a = m_a * n_a;

  }; // DiamagCavityParams

}; // struct DiamagCavityParams

} // namespace kalypsso

#endif // KALYPSSO_CORE_PROBLEMS_DIAMAG_CAVITY_PARAMS_H_
