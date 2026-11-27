// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorBitMaskArray.cpp
 */
#include <kalypsso/core/MirrorBitMaskArray.h>

#include <kalypsso/core/Box.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorBitMaskArray<dim, device_t>::update_mask(
  Lock,
  MeshMap<dim, device_t> &                           mesh_map,
  MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata,
  size_t                                             mask_id)
{
  const auto num_mirrors = mesh_map.get_amr_mesh_info().local_num_mirrors();

  auto config = m_mirror_bit_mask_configs_host(mask_id);

  auto mask_array = get_mirror_bit_mask(mask_id);

  const auto mirror_keys = mesh_map.mirror_orchard_keys();
  const auto keys = mesh_map.orchard_keys();


  if (config.type == +MirrorBitMaskArrayType::CELL_CENTER)
  {
    auto   num_bytes_per_mirror = config.num_bytes_per_mirror();
    size_t num_iterations = num_mirrors * num_bytes_per_mirror;

    Kokkos::parallel_for(
      "MirrorBitMaskArray<dim, device_t>::update_mask",
      Kokkos::RangePolicy<exec_space>(0, num_iterations),
      KOKKOS_LAMBDA(const size_t index) {
        uint8_t res = 0;

        // index = i_byte + num_bytes_per_mirror * i_mirror
        const auto i_mirror = index / num_bytes_per_mirror;
        const auto i_byte = index - i_mirror * num_bytes_per_mirror;

        const auto i_elem_offset = i_byte * 8;

        const auto mirror_key = mirror_keys(i_mirror);

        // process 8 by 8 elements
        for (size_t i_elem = i_elem_offset; i_elem < i_elem_offset + 8; ++i_elem)
        {
          if (i_elem < config.num_elements_per_mirror())
          {
            auto ijk = cell_index_unravel(i_elem, config.block_sizes);

            // algorithm:
            // construct box corresponding to current cell element
            // for all active neighbor quadrant
            //   - construct box corresponding to stencil ghost shape
            //   - if current cell box intersects neighbor box, set bit mask to 1

            const auto cell_box = Box<dim>(convert_coord<dim, int32_t, real_t>(ijk),
                                           make_kokkos_array<real_t, dim>(ONE_F));

            for (size_t i_neighbor = 0;
                 i_neighbor < MirrorQuadrantUtilsMetadata<dim, device_t>::NEIGHBOR_STENCIL_SIZE;
                 ++i_neighbor)
            {
              const auto direction = utils::flat_index_to_direction<dim>(i_neighbor);

              const auto index_neighbor =
                i_neighbor +
                MirrorQuadrantUtilsMetadata<dim, device_t>::NEIGHBOR_STENCIL_SIZE * i_mirror;
              if (mirror_quad_metadata.active_neighbor(index_neighbor))
              {
                const auto neighbor_i_oct =
                  mirror_quad_metadata.mirror_neighbor_index(i_mirror, direction);
                const auto neighbor_key = keys(neighbor_i_oct);
                //  const auto neighbor_box = make_neighbor_quadrant_ghost_box();
                auto neighbor_box =
                  Box<dim>(convert_coord<dim, int32_t, real_t>(config.shift),
                           convert_coord<dim, int32_t, real_t>(config.ghost_sizes));
                const auto level = orchard_key_t<dim>::level(mirror_key);
                const auto level_neigh = orchard_key_t<dim>::level(neighbor_key);
                if (level_neigh > level)
                {
                  neighbor_box.homothety(0.5);
                }
                else if (level_neigh < level)
                {
                  neighbor_box.homothety(2.0);
                }
                else
                {
                }
                // TODO
              }
            } // end for i_neighbor
          } // end if i_elem
        } // end for i_elem

        mask_array(index) = res;
      });
  }

} // MirrorBitMaskArray<dim, device_t>::update_mask

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorBitMaskArray<dim, device_t>::reset(size_t num_mirrors)
{
  // update offsets and sizes (on host)
  size_t accum = 0;
  m_mirror_bit_mask_configs_host(0).offset = 0;
  m_mirror_bit_mask_configs_host(0).size =
    m_mirror_bit_mask_configs_host(0).bit_mask_size_in_bytes(num_mirrors);

  for (size_t i_mask = 1; i_mask < m_num_mirror_bit_mask; ++i_mask)
  {
    m_mirror_bit_mask_configs_host(i_mask).offset =
      m_mirror_bit_mask_configs_host(i_mask - 1).offset +
      m_mirror_bit_mask_configs_host(i_mask - 1).size;
    m_mirror_bit_mask_configs_host(i_mask).size =
      m_mirror_bit_mask_configs_host(i_mask).bit_mask_size_in_bytes(num_mirrors);
  }

  // copy back configs on device
  Kokkos::deep_copy(m_mirror_bit_mask_configs, m_mirror_bit_mask_configs_host);

  // compute total size and resize storage
  size_t total_size = 0;
  for (size_t i_mask = 0; i_mask < m_num_mirror_bit_mask; ++i_mask)
  {
    total_size += m_mirror_bit_mask_configs_host(i_mask).size;
  }

  m_storage.resize(total_size);

} // MirrorBitMaskArray<dim, device_t>::recompute_mask_offsets

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
size_t
MirrorBitMaskArray<dim, device_t>::register_mirror_bit_mask(
  block_size_t<dim> const & block_sizes,
  shift_t<dim> const &      shift,
  block_size_t<dim> const & ghost_sizes,
  MirrorBitMaskArrayType    mirror_bit_mask_type)
{
  // get new mirror mirror bit mask id
  auto mirror_bit_mask_id = m_num_mirror_bit_mask;

  // create new config
  const auto config = MirrorBitMaskConfig<dim>(
    mirror_bit_mask_id, block_sizes, shift, ghost_sizes, mirror_bit_mask_type);

  // add a new config in array (on host)
  m_mirror_bit_mask_configs_host.resize(m_num_mirror_bit_mask + 1);
  m_mirror_bit_mask_configs_host(mirror_bit_mask_id) = config;

  m_num_mirror_bit_mask++;

  return mirror_bit_mask_id;

} // MirrorBitMaskArray<dim, device_t>::register_mirror_bit_mask

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
size_t
MirrorBitMaskArray<dim, device_t>::register_mirror_bit_mask(
  block_size_t<dim> const &      block_sizes,
  MirrorBitMaskArrayType const & mirror_bit_mask_type)
{
  // get new mirror mirror bit mask id
  auto mirror_bit_mask_id = m_num_mirror_bit_mask;

  // special case when dealing with a flux array
  const auto shift = get_shift<dim>(0);
  const auto ghost_sizes = get_block_size<dim>(0);

  return register_mirror_bit_mask(block_sizes, shift, ghost_sizes, mirror_bit_mask_type);

} // MirrorBitMaskArray<dim, device_t>::register_mirror_bit_mask

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorBitMaskArray<dim, device_t>::update_all_masks(
  MeshMap<dim, device_t> &                           mesh_map,
  MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata)
{
  // 1. resize configs array on device
  m_mirror_bit_mask_configs.resize(m_num_mirror_bit_mask);

  // 2.
  // - recompute all offsets and sizes (and update configs on device)
  // - copy configuration on device
  // - resize storage
  reset(mesh_map.get_amr_mesh_info().local_num_mirrors());

  // 3. recompute mask pattern
  for (size_t i_mask = 0; i_mask > m_num_mirror_bit_mask; ++i_mask)
  {
    update_mask(Lock{}, mesh_map, mirror_quad_metadata, i_mask);
  }

} // MirrorBitMaskArray<dim, device_t>::update_all_masks

} // namespace core

} // namespace kalypsso
