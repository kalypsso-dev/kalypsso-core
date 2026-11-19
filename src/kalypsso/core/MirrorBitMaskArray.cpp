// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorBitMaskArray.cpp
 */
#include <kalypsso/core/MirrorBitMaskArray.h>

#include <kalypsso/core/StencilHelper.h>

namespace kalypsso
{

namespace core
{

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

  // add a new config in array
  m_mirror_bit_mask_configs.resize(m_num_mirror_bit_mask + 1);
  m_mirror_bit_mask_configs(mirror_bit_mask_id) = config;

  m_num_mirror_bit_mask++;

  return mirror_bit_mask_id;
}

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

  const auto shift = get_shift<dim>(0);
  const auto ghost_sizes = get_block_size<dim>(0);

  return register_mirror_bit_mask(block_sizes, shift, ghost_sizes, mirror_bit_mask_type);
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
MirrorBitMaskArray<dim, device_t>::update_mask(
  MeshMap<dim, device_t> &                           mesh_map,
  MirrorQuadrantUtilsMetadata<dim, device_t> const & mirror_quad_metadata,
  size_t                                             mirror_bit_mask_id)
{
  const auto local_num_mirrors = mesh_map.get_amr_mesh_info().local_num_mirrors();

  auto config = m_mirror_bit_mask_configs(mirror_bit_mask_id);

  // allocated memory in storage
  const auto old_size = m_storage.size();
  const auto new_size = old_size + config.bit_mask_size_in_bytes(local_num_mirrors);

  m_storage.resize(new_size);
}

} // namespace core

} // namespace kalypsso
