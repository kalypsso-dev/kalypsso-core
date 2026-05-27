// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file overwrite_velocity_field.cpp
 */
#include <kalypsso/core/OverwriteVelocityField.h>

namespace kalypsso
{

namespace core
{

// ================================================================================================
// ================================================================================================
OverwriteVelocityFieldType
get_overwrite_velocity_field_type(ConfigMap const & config_map)
{
  const auto overwrite_velocity_field_type_name =
    config_map.getString("overwrite_velocity", "type", "UNIFORM");
  auto maybe_value =
    OverwriteVelocityFieldType::_from_string_nothrow(overwrite_velocity_field_type_name.c_str());
  if (maybe_value)
  {
    return *maybe_value;
  }
  else
  {
    Kokkos::abort("Wrong parameter for amr/overwrite_velocity_type.");
  }
  return OverwriteVelocityFieldType::INVALID;
}


// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
OverwriteVelocityField<dim, device_t>::run(ConfigMap const &          config_map,
                                           DataArrayBlock_t const &   Udata,
                                           int32_t                    local_num_octants,
                                           OrchardKeys const &        orchard_keys,
                                           OverwriteVelocityFieldType velocity_type,
                                           size_t                     i_rho,
                                           size_t                     i_etot,
                                           size_t                     i_u,
                                           size_t                     i_v,
                                           size_t                     i_w,
                                           [[maybe_unused]] real_t    t)
{
  const auto scaling_factor = get_scaling_factor(config_map);

  // compute total number of cells
  const auto total_num_cells = local_num_octants * Udata.num_cells();

  if (velocity_type == +OverwriteVelocityFieldType::UNIFORM)
  {
    const auto params = UniformVelocityParams<dim>(config_map);

    Kokkos::parallel_for(
      "OverwriteVelocityFieldType::UNIFORM",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
    // the only reason of the following dummy code to be here, is that cuda nvcc compile
    // doesn't support capturing variables inside the inner lambda inside a constexpr if
    //
    // strangely, nvc++ is ok and don't need that
    // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (i_w != i_w)
          dummy++;
#endif

        // convert global index into
        // - octant id
        // - cell_index inside block (from 0 to num_cells-1)
        const auto iOct = global_index / Udata.num_cells();
        const auto cell_index = global_index - iOct * Udata.num_cells();

        const auto rho = Udata(cell_index, i_rho, iOct);
        const auto etot_old = Udata(cell_index, i_etot, iOct);
        const auto rhou_old = Udata(cell_index, i_u, iOct);
        const auto rhov_old = Udata(cell_index, i_v, iOct);
        auto       rhow_old = ZERO_F;
        if constexpr (dim == 3)
          rhow_old = Udata(cell_index, i_w, iOct);

        auto ekin_old = HALF_F * (rhou_old * rhou_old + rhov_old * rhov_old) / rho;
        if constexpr (dim == 3)
          ekin_old += HALF_F * (rhow_old * rhow_old) / rho;

        real_t ekin_new = params.v[IX] * params.v[IX] + params.v[IY] * params.v[IY];
        if constexpr (dim == 3)
          ekin_new += params.v[IZ] * params.v[IZ];
        ekin_new = HALF_F * rho * ekin_new;

        // overwrite velocity field
        Udata(cell_index, i_u, iOct) = rho * params.v[IX];
        Udata(cell_index, i_v, iOct) = rho * params.v[IY];
        if constexpr (dim == 3)
          Udata(cell_index, i_w, iOct) = rho * params.v[IZ];

        // update total energy
        Udata(cell_index, i_etot, iOct) = etot_old - ekin_old + ekin_new;
      });
  }
  else if (velocity_type == +OverwriteVelocityFieldType::VORTEX)
  {
    const auto params = VortexVelocityParams<dim>(config_map);
    const auto xyz_min = get_xyz_min<dim>(config_map);

    Kokkos::parallel_for(
      "OverwriteVelocityFieldType::UNIFORM",
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
    // the only reason of the following dummy code to be here, is that cuda nvcc compile
    // doesn't support capturing variables inside the inner lambda inside a constexpr if
    //
    // strangely, nvc++ is ok and don't need that
    // TODO: remove theses lines when nvcc will have support for this.
#ifdef __NVCC__
        [[maybe_unused]] int dummy = 0;
        if (i_w != i_w)
          dummy++;
#endif

        // convert global index into
        // - octant id
        // - cell_index inside block (from 0 to num_cells-1)
        const auto    iOct = global_index / Udata.num_cells();
        const int32_t cell_index = static_cast<int32_t>(global_index - iOct * Udata.num_cells());

        const auto & block_sizes = Udata.block_size();

        // compute ix,iy,iz of local cell inside
        // block from index
        auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

        // get block orchard key
        const auto key = orchard_keys(iOct);

        // compute physical x,y,z for that cell (cell center)
        const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

        const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

        const auto rho = Udata(cell_index, i_rho, iOct);
        const auto etot_old = Udata(cell_index, i_etot, iOct);
        const auto rhou_old = Udata(cell_index, i_u, iOct);
        const auto rhov_old = Udata(cell_index, i_v, iOct);
        auto       rhow_old = ZERO_F;
        if constexpr (dim == 3)
          rhow_old = Udata(cell_index, i_w, iOct);

        auto ekin_old = HALF_F * (rhou_old * rhou_old + rhov_old * rhov_old) / rho;
        if constexpr (dim == 3)
          ekin_old += HALF_F * (rhow_old * rhow_old) / rho;

        const auto vx = params.vx(t, xyz);
        const auto vy = params.vy(t, xyz);
        real_t     ekin_new = vx * vx + vy * vy;
        if constexpr (dim == 3)
        {
          const auto vz = params.vz(t, xyz);
          ekin_new += vz * vz;
        }
        ekin_new = HALF_F * rho * ekin_new;

        // overwrite velocity field
        Udata(cell_index, i_u, iOct) = rho * params.vx(t, xyz);
        Udata(cell_index, i_v, iOct) = rho * params.vy(t, xyz);
        if constexpr (dim == 3)
          Udata(cell_index, i_w, iOct) = rho * params.vz(t, xyz);

        // update total energy
        Udata(cell_index, i_etot, iOct) = etot_old - ekin_old + ekin_new;
      });
  }

} // OverwriteVelocityField<dim, device_t>::run

template class OverwriteVelocityField<2, kalypsso::DefaultDevice>;
template class OverwriteVelocityField<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
