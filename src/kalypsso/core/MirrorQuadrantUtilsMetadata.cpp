// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorQuadrantUtilsMetadata.cpp
 */
#include <kalypsso/core/MirrorQuadrantUtilsMetadata.h>

namespace kalypsso
{

namespace core
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_FUNCTION auto
MirrorQuadrantUtilsMetadata<dim, device_t>::mirror_neighbor_index(iOct_t mirror_id,
                                                                  int8_t dir_x,
                                                                  int8_t dir_y) const -> iOct_t
{
  return m_mirror_index_to_neighbor_global_index(mirror_id * NEIGHBOR_STENCIL_SIZE +
                                                 flat_direction(dir_x, dir_y));
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_FUNCTION auto
MirrorQuadrantUtilsMetadata<dim, device_t>::mirror_neighbor_index(iOct_t mirror_id,
                                                                  int8_t dir_x,
                                                                  int8_t dir_y,
                                                                  int8_t dir_z) const -> iOct_t
{
  return m_mirror_index_to_neighbor_global_index(mirror_id * NEIGHBOR_STENCIL_SIZE +
                                                 flat_direction(dir_x, dir_y, dir_z));
}

// explicit template instantiation
template class MirrorQuadrantUtilsMetadata<2, kalypsso::DefaultDevice>;
template class MirrorQuadrantUtilsMetadata<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso
