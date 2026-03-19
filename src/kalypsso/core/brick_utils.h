// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file brick_utils.h
 */
#ifndef KALYPSSO_CORE_BRICKUTILS_H_
#define KALYPSSO_CORE_BRICKUTILS_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/enums.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/brick_base.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/p4est/connectivity.h> // for CONNECTIVITY_PERIODIC_FALSE

namespace kalypsso
{

// ===========================================================
// ===========================================================
//! Just return a dim-dimensional uniform array of boolean.
//! TODO: this function should probably be defined elsewhere.
template <size_t dim>
constexpr auto
get_bool_array(bool value)
{
  if constexpr (dim == 2)
  {
    return Kokkos::Array<bool, dim>{ value, value };
  }
  if constexpr (dim == 3)
  {
    return Kokkos::Array<bool, dim>{ value, value, value };
  }
} // get_bool_array

// // ===========================================================
// // ===========================================================
// //! small utility to transform brick_sizes (3d array) into an array of size dim
// template <typename T, size_t dim>
// auto
// get_brick_sizes(Kokkos::Array<int, dim> brick_sizes) -> Kokkos::Array<T, dim>
// {
//   if constexpr (dim == 2)
//   {
//     return Kokkos::Array<T, dim>{ static_cast<T>(brick_sizes[0]), static_cast<T>(brick_sizes[1])
//     };
//   }
//   if constexpr (dim == 3)
//   {
//     return Kokkos::Array<T, dim>{ static_cast<T>(brick_sizes[0]),
//                                   static_cast<T>(brick_sizes[1]),
//                                   static_cast<T>(brick_sizes[2]) };
//   }
// } // get_brick_sizes

// ===========================================================
// ===========================================================
//! small utility extract brick_sizes from config map.
template <size_t dim>
auto
get_brick_sizes(ConfigMap const & config_map) -> brick_size_t<dim>
{

  // brick sizes
  brick_size_t<dim> brick_sizes;

  using T = typename brick_size_t<dim>::value_type;

  brick_sizes[IX] = static_cast<T>(config_map.getInteger("p4est_connectivity", "nbrick_x", 1));
  brick_sizes[IY] = static_cast<T>(config_map.getInteger("p4est_connectivity", "nbrick_y", 1));
  if constexpr (dim == 3)
    brick_sizes[IZ] = static_cast<T>(config_map.getInteger("p4est_connectivity", "nbrick_z", 1));

  {
    // check that nbrick_x, nbrick_y and nbrick_z have values in valid range
    assertm(brick_sizes[IX] >= 0 and brick_sizes[IX] <= orchard_key_t<dim>::MAX_NB_TREES_PER_DIR,
            "nbrick_x has invalid value");
    assertm(brick_sizes[IY] >= 0 and brick_sizes[IY] <= orchard_key_t<dim>::MAX_NB_TREES_PER_DIR,
            "nbrick_y has invalid value");
    if constexpr (dim == 3)
      assertm(brick_sizes[IZ] >= 0 and brick_sizes[IZ] <= orchard_key_t<dim>::MAX_NB_TREES_PER_DIR,
              "nbrick_z has invalid value");
  }

  return brick_sizes;

} // get_brick_sizes

// ===========================================================
// ===========================================================
//! small utility extract brick periodicity from config map.
template <size_t dim>
auto
get_brick_periodicity(ConfigMap const & config_map) -> Kokkos::Array<bool, dim>

{
  // brick periodicty
  Kokkos::Array<bool, dim> res;

  res[IX] = static_cast<bool>(
    config_map.getInteger("p4est_connectivity", "periodic_x", CONNECTIVITY_PERIODIC_FALSE));
  res[IY] = static_cast<bool>(
    config_map.getInteger("p4est_connectivity", "periodic_y", CONNECTIVITY_PERIODIC_FALSE));

  if constexpr (dim == 3)
  {
    res[IZ] = static_cast<bool>(
      config_map.getInteger("p4est_connectivity", "periodic_z", CONNECTIVITY_PERIODIC_FALSE));
  }

  return res;

} // get_brick_periodicity

// ======================================================
// ======================================================
//! return lower left corner coordinates
template <size_t dim>
auto
lower_left_corner(ConfigMap const & config_map)
{

  const real_t xmin = config_map.getReal("mesh", "xmin", ZERO_F);
  const real_t ymin = config_map.getReal("mesh", "ymin", ZERO_F);
  const real_t zmin = config_map.getReal("mesh", "zmin", ZERO_F);

  if constexpr (dim == 2)
  {
    Kokkos::Array<real_t, 2> llc{ xmin, ymin };
    return llc;
  }
  else if constexpr (dim == 3)
  {
    Kokkos::Array<real_t, 3> llc{ xmin, ymin, zmin };
    return llc;
  }
}

// ======================================================
// ======================================================
//! same as lower left corner
template <size_t dim>
auto
get_xyz_min(ConfigMap const & config_map)
{
  return lower_left_corner<dim>(config_map);
}

// ======================================================
// ======================================================
//! return upper right corner coordinates
template <size_t dim>
auto
upper_right_corner(ConfigMap const & config_map)
{

  const auto xmin = config_map.getReal("mesh", "xmin", ZERO_F);
  const auto ymin = config_map.getReal("mesh", "ymin", ZERO_F);
  const auto zmin = config_map.getReal("mesh", "zmin", ZERO_F);

  const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
  const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
  const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

  const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", ONE_F);

  const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
  const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
  const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

  if constexpr (dim == 2)
  {
    Kokkos::Array<real_t, 2> urc{ xmax, ymax };
    return urc;
  }
  else if constexpr (dim == 3)
  {
    Kokkos::Array<real_t, 3> urc{ xmax, ymax, zmax };
    return urc;
  }
}

// ======================================================
// ======================================================
//! same as upper right corner
template <size_t dim>
auto
get_xyz_max(ConfigMap const & config_map)
{
  return upper_right_corner<dim>(config_map);
}

// ======================================================
// ======================================================
//! return a Kokkos::Array of scaling factor to apply for convert x,y,z coordinates from p4est
//! brick connectivity space to real space [x_min, x_max] x [y_min, y_max] x [z_min, z_max]
inline auto
get_scaling_factor(ConfigMap const & config_map)
{
  const real_t scaling_factor = config_map.getReal("mesh", "scaling_factor", ONE_F);

  return scaling_factor;
} // get_scaling_factor

} // namespace kalypsso

#endif // KALYPSSO_CORE_BRICKUTILS_H_
