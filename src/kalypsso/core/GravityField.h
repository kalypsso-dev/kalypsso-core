// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * GravityField.h
 *
 * Implementation notes:
 * - currently only uniform gravity is implemented, but other type of analytical or numerical
 * gravity field can be implemented by adding items in GravityType enum and providing implementation
 * in class GravityField
 *
 * - we don't use class inheritance and virtual member to implement different types of gravity
 * because virtual members are not supported in device code.
 *
 * A gravity field must provide member gx, gy, and gz to access grivity field component at location
 * (x,y,z).
 *
 */
#ifndef KALYPSSO_CORE_GRAVITYFIELD_H_
#define KALYPSSO_CORE_GRAVITYFIELD_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/enums.h>
#include <kalypsso/utils/config/ConfigMap.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

// clang-format off
BETTER_ENUM(GravityType, int, UNDEFINED, UNIFORM)
// clang-format on

GravityType
get_gravity_type(ConfigMap const & config_map);

// =======================================================================
template <size_t dim>
Kokkos::Array<real_t, dim>
get_uniform_gravity_vector(ConfigMap const & config_map)
{
  Kokkos::Array<real_t, dim> res;
  res[IX] = config_map.getReal("gravity", "uniform_gx", ZERO_F);
  res[IY] = config_map.getReal("gravity", "uniform_gy", ZERO_F);
  if constexpr (dim == 3)
  {
    res[IZ] = config_map.getReal("gravity", "uniform_gz", ZERO_F);
  }
  return res;
}

/**
 * TODO: remove this type alias when several types of gravity fields will be implemented.
 */
template <size_t dim>
using UniformGravityField = Kokkos::Array<real_t, dim>;

// ===============================================================================================
// ===============================================================================================
// ===============================================================================================
/**
 * Uniform gravity field.
 */
// template <size_t dim>
// class UniformGravityField
// {
// public:
//   // =============================================================
//   // =============================================================
//   UniformGravityField(ConfigMap const & config_map)
//     : m_g(get_uniform_gravity_vector<dim>(config_map))
//   {}

//   ~UniformGravityField() = default;

//   /**
//    * uniform gravity implementation
//    */
//   KOKKOS_INLINE_FUNCTION
//   real_t
//   gx(Kokkos::Array<real_t, dim> const & xyz) const
//   {
//     return m_g[IX];
//   }

//   KOKKOS_INLINE_FUNCTION
//   real_t
//   gy(Kokkos::Array<real_t, dim> const & xyz) const
//   {
//     return m_g[IY];
//   }

//   KOKKOS_INLINE_FUNCTION
//   real_t
//   gz(Kokkos::Array<real_t, dim> const & xyz) const
//   {
//     return [&]() {
//       if constexpr (dim == 2)
//         return 0;
//       else if constexpr (dim == 3)
//         return m_g[IZ];
//     }();
//   }

//   KOKKOS_INLINE_FUNCTION
//   Kokkos::Array<real_t, dim>
//   g(Kokkos::Array<real_t, dim> const & xyz) const
//   {
//     return m_g;
//   }

// private:
//   //! uniform gravity vector
//   Kokkos::Array<real_t, dim> m_g;

// }; // class UniformGravityField

} // namespace kalypsso

#endif // KALYPSSO_CORE_GRAVITYFIELD_H_
