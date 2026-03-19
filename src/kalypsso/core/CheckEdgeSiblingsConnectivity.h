// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CheckEdgeSiblingsConnectivity.h
 *
 *
 */
#ifndef KALYPSSO_CORE_CHECKEDGESIBLINGSCONNECTIVITY_H_
#define KALYPSSO_CORE_CHECKEDGESIBLINGSCONNECTIVITY_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>

#include <kalypsso/core/DataArrayBlock_utils.h>
#include <kalypsso/core/EdgeDataArrayBlock_utils.h>
#include <kalypsso/core/ConformalFaceStatus.h>

#include <type_traits>

namespace kalypsso
{

namespace core
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * \class CheckEdgeSiblingsConnectivity
 * \brief Check that edge siblings at block border are valid.
 *
 * The purpose of this functor is to check that give an edge location, all siblings edge locations
 * as computed by StencilHelper::getEdgeSiblingLoc are valid. This functor cross-check a posteriori
 * at run-time the functionality of StencilHelper (instead of writing an exhaustive unit test which
 * is quite complex).
 *
 * Let's remind that given an edge location, when requesting all edge siblings (i.e. edge locations
 * of all neighbor cells sharing the same edge), there are three possible situations:
 *
 * - either there are only two valid edge siblings; that happens when edge is at a non-conformal
 * interface and lies in the middle of a large face (of the coarse cell neighbor).
 * - either there are four valid edge siblings that maybe live at different AMR level:
 *   - all at the level "l"
 *   - some at level "l", some at level "l+1" (when there are cell neighbors at finer AMR level)
 *   - some at level "l", some at level "l-1" (when there is a coarser cell neighbor)
 *
 * The cross-check here simply consist in computing in physical space the coordinates of the edge
 * center and testing if we have the exact same value for all siblings.
 */
template <size_t dim, typename device_t>
class CheckEdgeSiblingsConnectivity
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using CellLocation_t = CellLocation<dim>;
  using EdgeLocation_t = EdgeLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  const AMRMeshInfo m_amr_mesh_info;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! edge flat index offsets
  const edge_flat_index_offset_t m_edge_flat_index_offsets;

  //! threshold
  KALYPSSO_STATIC_MATH_CONSTANT(SMALL, 1e-13);

public:
  /**
   * \param[in]  stencil_helper A stencil helper object.
   * \param[in]  orchard_keys Orchard keys.
   * \param[in]  amr_mesh_info AMR mesh info.
   * \param[in]  config_map A config map.
   *
   */
  CheckEdgeSiblingsConnectivity(block_size_t<dim>  bSize,
                                StencilHelper_t    stencil_helper,
                                orchard_key_view_t orchard_keys,
                                AMRMeshInfo        amr_mesh_info,
                                ConfigMap const &  config_map);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(block_size_t<dim>        bSize,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        AMRMeshInfo              amr_mesh_info,
        ConfigMap const &        config_map,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION bool
  are_location_not_near(Kokkos::Array<real_t, dim> const & xyz0,
                        Kokkos::Array<real_t, dim> const & xyz1) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  expect_near_xyz(EdgeLocation<dim> const &          edge_loc0,
                  Kokkos::Array<real_t, dim> const & xyz0,
                  Kokkos::Array<real_t, dim> const & xyz1) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  expect_near_same_level(EdgeLocation<dim> const & edge_loc0,
                         EdgeLocation<dim> const & edge_loc1) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  expect_near_finer_level(EdgeLocation<dim> const & edge_loc,
                          EdgeLocation<dim> const & edge_loc0) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  check(index_t const & edge_flatindex, int32_t const & iOct) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // CheckEdgeSiblingsConnectivity

// explicit template instantiation
extern template class CheckEdgeSiblingsConnectivity<2, kalypsso::DefaultDevice>;
extern template class CheckEdgeSiblingsConnectivity<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_CHECKEDGESIBLINGSCONNECTIVITY_H_
