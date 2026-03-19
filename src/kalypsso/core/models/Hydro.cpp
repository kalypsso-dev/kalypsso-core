// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Hydro.cpp
 */
#include <kalypsso/core/models/Hydro.h>

namespace kalypsso
{

namespace core
{

namespace models
{

// clang-format off
const id2names_t Hydro::m_id2names_all = {
  { ID, "rho" },
  { IE, "e_tot" },
  { IU, "rho_vx" },
  { IV, "rho_vy" },
  { IW, "rho_vz" },
  { IGX, "grav_x" },
  { IGY, "grav_y" },
  { IGZ, "grav_z" }
};
// clang-format on

} // namespace models

} // namespace core

} // namespace kalypsso
