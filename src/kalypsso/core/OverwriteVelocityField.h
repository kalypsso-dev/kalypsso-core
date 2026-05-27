// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file overwrite_velocity_field.h
 */
#ifndef KALYPSSO_CORE_OVERWRITE_VELOCITY_FIELD_H_
#define KALYPSSO_CORE_OVERWRITE_VELOCITY_FIELD_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost
#include <kalypsso/core/enums.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/core/MeshMap.h> // for orchard_key_view_t and amr_map_t
#include <kalypsso/core/orchard_key_utils.h>

#include <../../better-enums/enum.h>


namespace kalypsso
{

namespace core
{

// clang-format off
BETTER_ENUM(OverwriteVelocityFieldType, int, INVALID, UNIFORM, VORTEX)
// clang-format on

/**
 * Read configuration map to initialize the overwrite velocity type.
 */
OverwriteVelocityFieldType
get_overwrite_velocity_field_type(ConfigMap const & config_map);


template <size_t dim>
struct UniformVelocityParams
{
  Kokkos::Array<real_t, dim> v;

  UniformVelocityParams(ConfigMap const & config_map)
  {
    v[IX] = config_map.getReal("overwrite_velocity_uniform", "vx", KALYPSSO_NUM(0.0));
    v[IY] = config_map.getReal("overwrite_velocity_uniform", "vy", KALYPSSO_NUM(0.0));
    if constexpr (dim == 3)
    {
      v[IZ] = config_map.getReal("overwrite_velocity_uniform", "vz", KALYPSSO_NUM(0.0));
    }
  }
}; // struct UniformVelocityParams

template <size_t dim>
struct VortexVelocityParams
{
  real_t T;

  VortexVelocityParams(ConfigMap const & config_map)
  {
    T = config_map.getReal("overwrite_velocity_vortex", "T", KALYPSSO_NUM(4.0));
  }

  KOKKOS_INLINE_FUNCTION real_t
  vx(real_t t, Kokkos::Array<real_t, dim> const & xyz) const
  {
    return -cos(PI_F * t / T) * sin(PI_F * xyz[IX]) * sin(PI_F * xyz[IX]) * sin(2 * PI_F * xyz[IY]);
  }

  KOKKOS_INLINE_FUNCTION real_t
  vy(real_t t, Kokkos::Array<real_t, dim> const & xyz) const
  {
    return cos(PI_F * t / T) * sin(PI_F * xyz[IY]) * sin(PI_F * xyz[IY]) * sin(2 * PI_F * xyz[IX]);
  }

  KOKKOS_INLINE_FUNCTION real_t
  vz([[maybe_unused]] real_t t, [[maybe_unused]] Kokkos::Array<real_t, dim> const & xyz) const
  {
    return ZERO_F;
  }

}; // struct VortexVelocityParams

/**
 * Overwrite velocity of a conservative variable array.
 */
template <size_t dim, typename device_t>
struct OverwriteVelocityField
{
  //! Array of Orchard keys
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  //! Data alias (for conservative variable array)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  using ExecutionSpace = typename device_t::execution_space;

  /**
   * Overwrite velocity of a conservative variable array.
   *
   * \param[in] conservative variables array
   * \param[in] velocity_type identifier of a predefined velocity field to use
   * \param[in] i_rho index to where density is stored
   * \param[in] i_etot index to where total energy is stored
   * \param[in] i_u index to where rho*u is stored
   * \param[in] i_v index to where rho*v is stored
   * \param[in] i_w index to where rho*w is stored
   */
  static void
  run(ConfigMap const &          config_map,
      DataArrayBlock_t const &   Udata,
      int32_t                    local_num_octants,
      OrchardKeys const &        orchard_keys,
      OverwriteVelocityFieldType velocity_type,
      size_t                     i_rho,
      size_t                     i_etot,
      size_t                     i_u,
      size_t                     i_v,
      size_t                     i_w,
      real_t                     t);

}; // struct OverwriteVelocityField

extern template class OverwriteVelocityField<2, kalypsso::DefaultDevice>;
extern template class OverwriteVelocityField<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_OVERWRITE_VELOCITY_FIELD_H_
