// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside.cpp
 *
 * Implement border conditions (other than periodic) for Hydrodynamics.
 */

#include "FillOutside.h"

#include <kalypsso/core/MeshMap.h> // key_to_value

namespace kalypsso
{

namespace test
{

// ==============================================================================
// ==============================================================================
template <size_t dim, typename device_t, typename AnalyticalBC>
FillOutsideCellFunctor<dim, device_t, AnalyticalBC>::FillOutsideCellFunctor(
  ConfigMap const &        config_map,
  amr_hashmap_t            amr_hashmap,
  orchard_key_view_t       orchard_keys,
  AMRMeshInfo              amr_mesh_info,
  DataArrayBlock_t         userdata,
  block_size_t<dim>        block_sizes,
  brick_size_t<dim>        brick_sizes,
  Kokkos::Array<bool, dim> is_brick_periodic,
  bc_array_t               bc_types,
  AnalyticalBC const &     f)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_userdata(userdata)
  , m_block_sizes(block_sizes)
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
  , m_bc_types(bc_types)
  , m_f(f)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
{}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t, typename AnalyticalBC>
void
FillOutsideCellFunctor<dim, device_t, AnalyticalBC>::apply(
  ConfigMap const &        config_map,
  amr_hashmap_t            amr_hashmap,
  orchard_key_view_t       orchard_keys,
  AMRMeshInfo              amr_mesh_info,
  DataArrayBlock_t         userdata,
  block_size_t<dim>        block_sizes,
  brick_size_t<dim>        brick_sizes,
  Kokkos::Array<bool, dim> is_brick_periodic,
  bc_array_t               bc_types)
{

  // create compute functor
  FillOutsideCellFunctor<dim, device_t> functor(config_map,
                                                amr_hashmap,
                                                orchard_keys,
                                                amr_mesh_info,
                                                userdata,
                                                block_sizes,
                                                brick_sizes,
                                                is_brick_periodic,
                                                bc_types);

  const auto nbCellsPerLeaf = userdata.num_cells();

  // only count outside leaves (beyond external border)
  const auto totalNumberOfOutsideCells =
    nbCellsPerLeaf * amr_mesh_info.total_local_number_of_outside_quads();

  Kokkos::parallel_for(
    "FillOutsideFunctor", Kokkos::RangePolicy<exec_space>(0, totalNumberOfOutsideCells), functor);

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t, typename AnalyticalBC>
void
FillOutsideCellFunctor<dim, device_t, AnalyticalBC>::apply(
  ConfigMap const &        config_map,
  amr_hashmap_t            amr_hashmap,
  orchard_key_view_t       orchard_keys,
  AMRMeshInfo              amr_mesh_info,
  DataArrayBlock_t         userdata,
  block_size_t<dim>        block_sizes,
  brick_size_t<dim>        brick_sizes,
  Kokkos::Array<bool, dim> is_brick_periodic,
  bc_array_t               bc_types,
  const AnalyticalBC &     f)
{
  // create compute functor
  FillOutsideCellFunctor<dim, device_t, AnalyticalBC> functor(config_map,
                                                              amr_hashmap,
                                                              orchard_keys,
                                                              amr_mesh_info,
                                                              userdata,
                                                              block_sizes,
                                                              brick_sizes,
                                                              is_brick_periodic,
                                                              bc_types,
                                                              f);

  const auto nbCellsPerLeaf = userdata.num_cells();

  // only count outside leaves (beyond external border)
  const auto totalNumberOfOutsideCells =
    nbCellsPerLeaf * amr_mesh_info.total_local_number_of_outside_quads();

  Kokkos::parallel_for(
    "FillOutsideFunctor", Kokkos::RangePolicy<exec_space>(0, totalNumberOfOutsideCells), functor);

} // apply - analytical border conditions

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t, typename AnalyticalBC>
KOKKOS_INLINE_FUNCTION void
FillOutsideCellFunctor<dim, device_t, AnalyticalBC>::operator()(const index_t & global_index) const
{

  const auto nbCellsPerLeaf = m_userdata.num_cells();

  // iOct_local, by design, is associated to an outside quadrant
  const auto iOct_outside =
    global_index / nbCellsPerLeaf + m_amr_mesh_info.first_outside_quad_local_id();

  const auto cellindex_out = global_index - nbCellsPerLeaf * (global_index / nbCellsPerLeaf);
  const auto coord_out = cellindex_to_coord<dim>(cellindex_out, m_block_sizes);

  // get orchard key corresponding to visited outside quadrants
  const auto key_outside = m_orchard_keys_device(iOct_outside);

  // when computing inside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal at the periodic image of key_outside
  auto outside_normal =
    orchard_key_t<dim>::get_outside_normal(key_outside, m_brick_sizes, is_periodic);

  outside_normal[IX] = outside_normal[IX] * orchard_key_t<dim>::is_touching_face_X(key_outside);
  outside_normal[IY] = outside_normal[IY] * orchard_key_t<dim>::is_touching_face_Y(key_outside);
  if constexpr (dim == 3)
  {
    outside_normal[IZ] = outside_normal[IZ] * orchard_key_t<dim>::is_touching_face_Z(key_outside);
  }

  // compute corresponding inside quadrant (just across external border, i.e. along outside
  // normal, but opposite direction)
  auto key_inside = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_outside, outside_normal, m_brick_sizes, m_is_brick_periodic);
  orchard_key_t<dim>::reset_outside_bits(key_inside);

  // get iOct_inside from the unordered map
  const auto key_status = key_to_value(key_inside, m_amr_hashmap_device);

  [[maybe_unused]] auto const & is_valid_key = key_status.first;

  // make sure key_inside actually exist in map
  // if it doesn't we have a serious logical problem
  KOKKOS_ASSERT(is_valid_key &&
                "[kalypsso::FillOutsideCellFunctor] key_inside does not exist in hashmap !?");

  const auto iOct_inside = key_status.second;

  // make sure iOct_inside is actually inside
  KOKKOS_ASSERT(
    iOct_inside < m_amr_mesh_info.first_outside_quad_local_id() &&
    "[kalypsso::FillOutsideCellFunctor] iOct does not identify an octant inside domain ?!");

  //
  // now we can fill outside userdata according to border condition
  //

  // get list of faces
  const auto faces = Face::get_all_faces<dim>();

  for (const auto face : faces)
  {
    // normal direction
    const Dir::dir_t dir = face / 2;

    if (orchard_key_t<dim>::is_at_domain_border(key_inside, face, m_brick_sizes) and
        orchard_key_t<dim>::is_outside_dir(key_outside, dir))
    {
      if (m_bc_types[face]._to_integral() == +BC_HYDRO::ZERO_GRADIENT)
      {
        auto coord_in = coord_out;
        coord_in[dir] = Face::is_left_face(face) ? 0 : m_block_sizes[dir] - 1;
        auto cellindex_in = coord_to_cellindex<dim>(coord_in, m_block_sizes);

        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
          m_userdata(cellindex_out, ivar, iOct_outside) =
            m_userdata(cellindex_in, ivar, iOct_inside);
      }
      else if (m_bc_types[face]._to_integral() == +BC_HYDRO::WALL)
      {
        auto coord_in = coord_out;

        coord_in[dir] = m_block_sizes[dir] - 1 - coord_out[dir];

        auto cellindex_in = coord_to_cellindex<dim>(coord_in, m_block_sizes);

        // copy data and negate normal velocity
        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          real_t scale = KALYPSSO_NUM(1.0);
          if (ivar == (Hydro::IU + dir))
          {
            scale = KALYPSSO_NUM(-1.0);
          }
          m_userdata(cellindex_out, ivar, iOct_outside) =
            scale * m_userdata(cellindex_in, ivar, iOct_inside);
        } // end for ivar

      } // end BC_HYDRO::WALL
      else if (m_bc_types[face]._to_integral() == +BC_HYDRO::ANALYTICAL)
      {

        const auto xyz_corner = outside_key_to_vertex_coord<dim>(key_outside, false, m_brick_sizes);

        const auto xyz_cell_vertex = compute_cell_coordinates<dim>(
          orchard_key_t<dim>::level(key_outside), xyz_corner, coord_out, m_block_sizes);

        const auto xyz_cell =
          vertex_coord_to_real_space<dim>(xyz_cell_vertex, m_scaling_factor, m_xyz_min);

        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          if constexpr (dim == 2)
          {
            m_userdata(cellindex_out, ivar, iOct_outside) = m_f(xyz_cell[IX], xyz_cell[IY], ivar);
          }
          else if constexpr (dim == 3)
          {
            m_userdata(cellindex_out, ivar, iOct_outside) =
              m_f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar);
          }
        }
      } // end BC_HYDRO::WALL
    } // end if is_at_domain_border
  } // for faces

} // operator()

// explicit template instantiation
template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice>;
template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice>;
template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice, InitFunc1>;
template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice, InitFunc1>;
template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice, InitFunc2>;
template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice, InitFunc2>;

} // namespace test

} // namespace kalypsso
