// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplMP.cpp
 */

#include <kalypsso/core/UserDataRemapperImplMD.h>
#include <kalypsso/core/FillBlockGhosts_common.h> // for compute_face_neighbor_key_finer

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
UserDataRemapperImplMD<dim, device_t>::UserDataRemapperImplMD(DataArrayBlockMultiVar_t old_data,
                                                              MaterialPresenceView_t   old_mat,
                                                              DataArrayBlockMultiVar_t new_data,
                                                              MaterialPresenceView_t   new_mat,
                                                              AmrHashmap_t  amr_hashmap_device_old,
                                                              OrchardKeys_t orchard_keys_device_old,
                                                              OrchardKeys_t orchard_keys_device_new,
                                                              const uint32_t    num_vars_per_mat,
                                                              const ConfigMap & config_map)
  : m_old_data(old_data)
  , m_old_mat(old_mat)
  , m_new_data(new_data)
  , m_new_mat(new_mat)
  , m_amr_hashmap_device_old(amr_hashmap_device_old)
  , m_orchard_keys_device_old(orchard_keys_device_old)
  , m_orchard_keys_device_new(orchard_keys_device_new)
  , m_stencil_helper(amr_hashmap_device_old,
                     orchard_keys_device_old,
                     old_data.shape(),
                     get_brick_sizes<dim>(config_map),
                     get_brick_periodicity<dim>(config_map))
  , m_prolongation(get_cell_prolongation_type(config_map))
  , m_num_vars_per_mat(static_cast<int32_t>(num_vars_per_mat))
  , m_block_sizes(old_data.shape())
{}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
KOKKOS_FUNCTION void
UserDataRemapperImplMD<dim, device_t>::operator()(const int32_t i_new) const
{
  const auto num_cells = m_new_data.num_cells();

  const auto i_new_oct = i_new / num_cells;
  const auto i_new_cell = i_new - i_new_oct * num_cells;

  const auto nbmat = m_new_mat.num_materials(i_new_oct);

  const auto new_key = m_orchard_keys_device_new(i_new_oct);
  const auto new_key_index = m_amr_hashmap_device_old.find(new_key);

  // It hasn't changed
  if (m_amr_hashmap_device_old.valid_at(new_key_index))
  {
    const auto i_old_oct = static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_key_index));
    const auto i_old_cell = i_new_cell;

    for (int32_t imat = 0; imat < nbmat; ++imat)
    {
      const auto mat_num = m_new_mat.material_num(i_new_oct, imat);

      if (m_old_mat.get(i_old_oct, mat_num)) // Is the material present in the source octant?
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto old_ivar =
            ivar + m_num_vars_per_mat * m_old_mat.material_index(i_old_oct, mat_num);
          const auto new_ivar = ivar + m_num_vars_per_mat * imat;

          m_new_data(i_new_cell, new_ivar, i_new_oct) = m_old_data(i_old_cell, old_ivar, i_old_oct);
        }
      else
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto new_ivar = ivar + m_num_vars_per_mat * imat;
          m_new_data(i_new_cell, new_ivar, i_new_oct) = 0;
        }
    }

    return;
  }

  const auto new_father_key = orchard_key_t<dim>::father(new_key);
  const auto new_father_key_index = m_amr_hashmap_device_old.find(new_father_key);

  // It was refined
  if (m_amr_hashmap_device_old.valid_at(new_father_key_index))
  {
    const auto i_old_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_father_key_index));
    const auto child_id = static_cast<int32_t>(orchard_key_t<dim>::family_id(new_key));

    const auto   i_new_coord = cellindex_to_coord<dim>(i_new_cell, m_block_sizes);
    coord_t<dim> i_old_coord;

    i_old_coord[IX] = i_new_coord[IX] / 2 + ((child_id & 0b001) >> 0) * (m_block_sizes[IX] / 2);
    i_old_coord[IY] = i_new_coord[IY] / 2 + ((child_id & 0b010) >> 1) * (m_block_sizes[IY] / 2);
    if constexpr (dim == 3)
      i_old_coord[IZ] = i_new_coord[IZ] / 2 + ((child_id & 0b100) >> 2) * (m_block_sizes[IZ] / 2);

    const auto i_old_cell = coord_to_cellindex<dim>(i_old_coord, m_block_sizes);

    if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      for (int32_t imat = 0; imat < nbmat; ++imat)
      {
        const auto mat_num = m_new_mat.material_num(i_new_oct, imat);

        if (m_old_mat.get(i_old_oct, mat_num)) // Is the material present in the source octant?
          for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
          {
            const auto old_ivar =
              ivar + m_num_vars_per_mat * m_old_mat.material_index(i_old_oct, mat_num);
            const auto new_ivar = ivar + m_num_vars_per_mat * imat;

            m_new_data(i_new_cell, new_ivar, i_new_oct) =
              m_old_data(i_old_cell, old_ivar, i_old_oct);
          }
        else
          for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
          {
            const auto out_ivar = ivar + m_num_vars_per_mat * imat;
            m_new_data(i_new_cell, out_ivar, i_new_oct) = 0;
          }
      }
    else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
    {
      const CellLocation<dim> cell_loc{ i_old_coord, new_father_key, i_old_oct, false };
      linear_extrapolate_using_limited_slopes(cell_loc, i_new_coord, i_new_cell, i_new_oct);
    }

    return;
  }

  // This means it was coarsened so we have to regroup all of the material of its children
  const auto new_child_key = orchard_key_t<dim>::eldest_child(new_key);
  const auto new_child_key_index = m_amr_hashmap_device_old.find(new_child_key);

  // Must be valid
  if (m_amr_hashmap_device_old.valid_at(new_child_key_index))
  {
    const auto i_eldest_child_oct =
      static_cast<int32_t>(m_amr_hashmap_device_old.value_at(new_child_key_index));
    const auto i_new_coord = cellindex_to_coord<dim>(i_new_cell, m_block_sizes);

    coord_t<dim> i_old_coord = 2 * i_new_coord;
    uint8_t      i_child = 0;
    if (i_new_coord[IX] >= m_block_sizes[IX] / 2)
    {
      i_old_coord[IX] -= m_block_sizes[IX];
      i_child |= 0x1;
    }
    if (i_new_coord[IY] >= m_block_sizes[IY] / 2)
    {
      i_old_coord[IY] -= m_block_sizes[IY];
      i_child |= 0x2;
    }
    if constexpr (dim == 3)
      if (i_new_coord[IZ] >= m_block_sizes[IZ] / 2)
      {
        i_old_coord[IZ] -= m_block_sizes[IZ];
        i_child |= 0x4;
      }

    const auto i_old_oct = i_eldest_child_oct + i_child;
    const auto i_old_key = m_orchard_keys_device_old(i_old_oct);

    for (int32_t imat = 0; imat < nbmat; ++imat)
    {
      const auto mat_num = m_new_mat.material_num(i_new_oct, imat);

      if (m_old_mat.get(i_old_oct, mat_num)) // Is the material present in the source octant?
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto new_ivar = ivar + m_num_vars_per_mat * imat;
          const auto old_ivar =
            ivar + m_num_vars_per_mat * m_old_mat.material_index(i_old_oct, mat_num);

          const CellLocation<dim> cell_loc{ i_old_coord, i_old_key, i_old_oct, false };
          m_new_data(i_new_cell, new_ivar, i_new_oct) = m_stencil_helper.compute_siblings_average(
            cell_loc, m_block_sizes, old_ivar, m_old_data);
        }
      else
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto new_ivar = ivar + m_num_vars_per_mat * imat;
          m_new_data(i_new_cell, new_ivar, i_new_oct) = 0;
        }
    }

    return;
  }
}

// ================================================================================================
// ================================================================================================
template class UserDataRemapperImplMD<2, kalypsso::DefaultDevice>;
template class UserDataRemapperImplMD<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
template class UserDataRemapperImplMD<2, HostDevice>;
template class UserDataRemapperImplMD<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

// ================================================================================================
// ================================================================================================
// INNER FUNCTIONS ADAPTED FROM UserDataRemapperImplBCC
// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplMD<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const & cell_loc_old,
  coord_t<2> const &      coord_new,
  int32_t const &         i_new_cell,
  int32_t const &         i_new_oct) const
{
  const real_t slope_type = 1;

  // get neighbor location, each of then may be at finer, same or coarser level compared
  // cell_loc_old
  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current new cell inside old parent cell using integer
  // coordinates in -1, +1
  const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;

  const auto nbmat = m_new_mat.num_materials(i_new_oct);

  for (int32_t imat = 0; imat < nbmat; ++imat)
  {
    // Extrapolation needs variable at each face so we check if it is present across each face.
    const auto mat_num = m_new_mat.material_num(i_new_oct, imat);
    const bool has_mat = m_old_mat.get(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_old.iOct), mat_num);

    if (has_mat) // Is the material present in the source octant?
      for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
      {
        const auto old_ivar =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_old.iOct), mat_num);

        const auto old_ivar_left_x =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num);

        const auto old_ivar_right_x =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num);

        const auto old_ivar_left_y =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num);

        const auto old_ivar_right_y =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num);

        const auto new_ivar = ivar + m_num_vars_per_mat * imat;

        // compute limited slopes
        auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(cell_loc_old,
                                                                              old_ivar,
                                                                              cell_loc_right_x,
                                                                              old_ivar_right_x,
                                                                              cell_loc_left_x,
                                                                              old_ivar_left_x,
                                                                              m_old_data,
                                                                              slope_type);

        auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(cell_loc_old,
                                                                              old_ivar,
                                                                              cell_loc_right_y,
                                                                              old_ivar_right_y,
                                                                              cell_loc_left_y,
                                                                              old_ivar_left_y,
                                                                              m_old_data,
                                                                              slope_type);

        // extrapolate
        m_new_data(i_new_cell, new_ivar, i_new_oct) =
          m_old_data(cell_loc_old.cellindex(m_block_sizes), old_ivar, cell_loc_old.iOct) +
          ONE_FOURTH_F * static_cast<real_t>(ix) * dudx +
          ONE_FOURTH_F * static_cast<real_t>(iy) * dudy;
      }
    else // If not, set the value to 0 (maybe best to switch extrapolation method?)
      for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
      {
        const auto new_ivar = ivar + m_num_vars_per_mat * imat;
        m_new_data(i_new_cell, new_ivar, i_new_oct) = 0;
      }
  }
}

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
UserDataRemapperImplMD<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const & cell_loc_old,
  coord_t<3> const &      coord_new,
  int32_t const &         i_new_cell,
  int32_t const &         i_new_oct) const
{
  const real_t slope_type = 1;

  // get neighbor location, each of then may be at finer, same or coarser level compared
  // cell_loc_old
  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_old, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current new cell inside old parent cell using integer
  // coordinates in -1, +1
  const int ix = 2 * (coord_new[IX] - 2 * (coord_new[IX] / 2)) - 1;
  const int iy = 2 * (coord_new[IY] - 2 * (coord_new[IY] / 2)) - 1;
  const int iz = 2 * (coord_new[IZ] - 2 * (coord_new[IZ] / 2)) - 1;

  const auto nbmat = m_new_mat.num_materials(i_new_oct);

  for (int32_t imat = 0; imat < nbmat; ++imat)
  {
    // Extrapolation needs variable at each face so we check if it is present across each face.
    const auto mat_num = m_new_mat.material_num(i_new_oct, imat);
    const bool has_mat = m_old_mat.get(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_left_z.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_right_z.iOct), mat_num) &&
                         m_old_mat.get(static_cast<int32_t>(cell_loc_old.iOct), mat_num);

    if (has_mat) // Is the material present in the source octant?
      for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
      {
        const auto old_ivar =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_old.iOct), mat_num);

        const auto old_ivar_left_x =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num);

        const auto old_ivar_right_x =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num);

        const auto old_ivar_left_y =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num);

        const auto old_ivar_right_y =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num);

        const auto old_ivar_left_z =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_left_z.iOct), mat_num);

        const auto old_ivar_right_z =
          ivar + m_num_vars_per_mat *
                   m_old_mat.material_index(static_cast<int32_t>(cell_loc_right_z.iOct), mat_num);

        const auto new_ivar = ivar + m_num_vars_per_mat * imat;

        // compute limited slopes
        auto const dudx = m_stencil_helper.compute_minmod_slopes_prolongation(cell_loc_old,
                                                                              old_ivar,
                                                                              cell_loc_right_x,
                                                                              old_ivar_right_x,
                                                                              cell_loc_left_x,
                                                                              old_ivar_left_x,
                                                                              m_old_data,
                                                                              slope_type);

        auto const dudy = m_stencil_helper.compute_minmod_slopes_prolongation(cell_loc_old,
                                                                              old_ivar,
                                                                              cell_loc_right_y,
                                                                              old_ivar_right_y,
                                                                              cell_loc_left_y,
                                                                              old_ivar_left_y,
                                                                              m_old_data,
                                                                              slope_type);

        auto const dudz = m_stencil_helper.compute_minmod_slopes_prolongation(cell_loc_old,
                                                                              old_ivar,
                                                                              cell_loc_right_z,
                                                                              old_ivar_right_z,
                                                                              cell_loc_left_z,
                                                                              old_ivar_left_z,
                                                                              m_old_data,
                                                                              slope_type);

        // extrapolate
        m_new_data(i_new_cell, new_ivar, i_new_oct) =
          m_old_data(cell_loc_old.cellindex(m_block_sizes), old_ivar, cell_loc_old.iOct) +
          ONE_FOURTH_F * static_cast<real_t>(ix) * dudx +
          ONE_FOURTH_F * static_cast<real_t>(iy) * dudy +
          ONE_FOURTH_F * static_cast<real_t>(iz) * dudz;
      }
    else // If not, set the value to 0 (maybe best to switch extrapolation method?)
      for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
      {
        const auto new_ivar = ivar + m_num_vars_per_mat * imat;
        m_new_data(i_new_cell, new_ivar, i_new_oct) = 0;
      }
  }
}

} // namespace kalypsso
