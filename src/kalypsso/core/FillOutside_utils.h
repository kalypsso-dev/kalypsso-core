// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside_utils.h
 *
 * Helper routines for filling blocks outside of physical domain (border conditions)
 * that can be re-used in different applications
 */
#ifndef KALYPSSO_CORE_FILLOUTSIDE_UTILS_H_
#define KALYPSSO_CORE_FILLOUTSIDE_UTILS_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/mesh_utils.h> // for struct Face
#include <kalypsso/core/real_type.h>

#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <iostream>

namespace kalypsso
{

//! return the total number of border regions
template <size_t dim>
constexpr auto
NUM_BORDERS()
{
  return Face::num_faces<dim>(); // + Edge::num_edges<dim>() + Corner::num_corners<dim>();
}

// =============================================================================
// =============================================================================
// pointwise init functor
struct AnalyticalZeroBC
{
  KOKKOS_FUNCTION auto
  operator()([[maybe_unused]] real_t x, [[maybe_unused]] real_t y, [[maybe_unused]] int var) const
  {
    return ZERO_F;
  }

  KOKKOS_FUNCTION auto
  operator()([[maybe_unused]] real_t x,
             [[maybe_unused]] real_t y,
             [[maybe_unused]] real_t z,
             [[maybe_unused]] int    var) const
  {
    return ZERO_F;
  }
};

// ===========================================================================
// ===========================================================================
//!
//! This struct does nothing except defining an enum of all possible border conditions
//!
//! \tparam BC must be a (better enum)
//!
template <class BC>
struct BorderConditionsConfig
{

  //! type alias to specify border conditions (other than periodic)
  template <size_t dim>
  using bc_array_t = Kokkos::Array<BC, NUM_BORDERS<dim>()>;

  // ==============================================================================
  // ==============================================================================
  template <size_t dim>
  static bc_array_t<dim>
  default_init()
  {
    if constexpr (dim == 2)
    {
      return { BC::PERIODIC, BC::PERIODIC, BC::PERIODIC, BC::PERIODIC };
    }
    else if constexpr (dim == 3)
    {
      return { BC::PERIODIC, BC::PERIODIC, BC::PERIODIC, BC::PERIODIC, BC::PERIODIC, BC::PERIODIC };
    }
  } // default_init

  // ==============================================================================
  // ==============================================================================
  template <size_t dim>
  static bc_array_t<dim>
  read_border_condition(ConfigMap const & config_map, ParallelEnv const & par_env)
  {
    // default value is periodic
    bc_array_t<dim> bc_types = default_init<dim>();

    // boundary_location_str can be "boundary_type_xmin", "boundary_type_xmax", ...
    auto read_bc = [&](std::string const & boundary_location_str) -> BC {
      auto bc_type_str = config_map.getString("mesh", boundary_location_str, "");

      // check if bc_type_str is a valid value
      auto maybe_bc = BC::_from_string_nothrow(bc_type_str.c_str());
      if (!maybe_bc)
      {
        // the string is not recognized or invalid
        if (par_env.rank() == 0)
        {
          std::cout << "[BorderConditionsConfig::read_border_condition] " << "\"" << bc_type_str
                    << "\"" << " is invalid (Using BC::NONE instead).\n"
                    << "Check your input parameter file.\n"
                    << "Valid values are:\n";
          for (const char * name : BC::_names())
            std::cout << name << " ";
          std::cout << "\n";
        }
        return BC::NONE;
      }
      else
      {
        return *maybe_bc;
      }
    };

    bc_types[XMIN] = read_bc("boundary_type_xmin");
    bc_types[XMAX] = read_bc("boundary_type_xmax");
    bc_types[YMIN] = read_bc("boundary_type_ymin");
    bc_types[YMAX] = read_bc("boundary_type_ymax");
    if constexpr (dim == 3)
    {
      bc_types[ZMIN] = read_bc("boundary_type_zmin");
      bc_types[ZMAX] = read_bc("boundary_type_zmax");
    }

    return bc_types;

  } // read_border_condition

}; // struct BorderConditionsConfig

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLOUTSIDE_UTILS_H_
