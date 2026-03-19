// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHD.cpp
 */
#include <kalypsso/core/models/MHD.h>

namespace kalypsso
{

namespace core
{

namespace models
{

// clang-format off
const id2names_t MHD::m_id2names_all = {
  { ID, "rho" },
  { IE, "e_tot" },
  { IU, "rho_vx" },
  { IV, "rho_vy" },
  { IW, "rho_vz" },
  { IBX, "Bx" },
  { IBY, "By" },
  { IBZ, "Bz" },
  { IA0, "B0x" },
  { IB0, "B0y" },
  { IC0, "B0z" },
  { IGX, "grav_x" },
  { IGY, "grav_y" },
  { IGZ, "grav_z" }
};
// clang-format on

} // namespace models

} // namespace core

} // namespace kalypsso
