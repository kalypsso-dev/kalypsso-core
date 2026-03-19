// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeGhostedScalarField.h
 */
#ifndef KALYPSSO_CORE_COMPUTEGHOSTEDSCALARFIELD_H_
#define KALYPSSO_CORE_COMPUTEGHOSTEDSCALARFIELD_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/prolongation.h>

namespace kalypsso
{

/**
 * \class ComputeGhostedScalarField
 *
 * This class is a direct adaptation of FillBlockGhostsFunctor:
 * - given in input a DataArrayBlock, extract one scalar field and fill ghost cells around all
 * blocks
 * - to make sure MPI ghost block are properly filled, one needs to compute first the mirror octant
 * (in the sens defined by p4est) and perform a MPI communication (this must be done in the calling
 * routine).
 *
 * \note No prolongation operator is applied to fill ghost cells when the neighbor block is at
 * coarser AMR level.
 *
 */
template <size_t dim, typename device_t>
class ComputeGhostedScalarField
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard keys that are "mirrors" (in the p4est sense).
  //! only used when we want to solely computed mirror quadrants.
  orchard_key_view_t m_mirror_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! starting octant id.
  //! this is a global octant id offset to the first octant to be processed when processing a group
  //! of owned quadrants.
  //!
  //! \note it is not used when processing mirrors
  //! quadrants, because mirror quadrants are process all at once.
  const int32_t m_iOct_begin;

  //! cell-centered variables (no ghosts, sizes= block_x,block_y,block_z)
  DataArrayBlock_t m_userdata_in;

  //! a ghosted data array (which block ghost cells need to be filled) - scalar field (only one
  //! variable)
  DataArrayGhostedBlock_t m_userdata_out;

  //! var id to be extracted from the input userdata
  int32_t m_ivar;

  //! prolongation type
  const CellCenteredProlongationType m_prolongation;

public:
  struct TagComputeMirrorQuad
  {};
  struct TagComputeAllQuad
  {};

  /**
   * Copy and fill ghost cells in a group of owned quadrants.
   *
   * \param[in] stencil helper
   * \param[in] amr_mesh_info number of octants (owned, ghost, outside, ...)
   * \param[in] iOct_begin is the first octant to process
   * \param[in] userdata_in data array used to fill ghost of userdata_out
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   *
   */
  ComputeGhostedScalarField(StencilHelper_t              stencil_helper,
                            AMRMeshInfo                  amr_mesh_info,
                            int32_t                      iOct_begin,
                            DataArrayBlock_t             userdata_in,
                            DataArrayGhostedBlock_t      userdata_out,
                            int32_t                      ivar,
                            CellCenteredProlongationType prolongation);

  //! same as above, but specifying also the mirror keys array
  ComputeGhostedScalarField(StencilHelper_t              stencil_helper,
                            orchard_key_view_t           mirror_orchard_keys,
                            AMRMeshInfo                  amr_mesh_info,
                            DataArrayBlock_t             userdata_in,
                            DataArrayGhostedBlock_t      userdata_out,
                            int32_t                      ivar,
                            CellCenteredProlongationType prolongation);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when processing in a group of octant
  static void
  apply_on_group(amr_hashmap_t            amr_hashmap,
                 orchard_key_view_t       orchard_keys,
                 AMRMeshInfo              amr_mesh_info,
                 int32_t                  iOct_begin,
                 int32_t                  num_octants_in_group,
                 DataArrayBlock_t         userdata_in,
                 DataArrayGhostedBlock_t  userdata_out,
                 int32_t                  ivar,
                 brick_size_t<dim>        brick_sizes,
                 Kokkos::Array<bool, dim> is_brick_periodic);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy.
  //!
  //! Use this member when processing only in mirror quadrants.
  static void
  apply_in_mirrors(amr_hashmap_t            amr_hashmap,
                   orchard_key_view_t       orchard_keys,
                   orchard_key_view_t       mirror_orchard_keys,
                   AMRMeshInfo              amr_mesh_info,
                   DataArrayBlock_t         userdata_in,
                   DataArrayGhostedBlock_t  userdata_out,
                   int32_t                  ivar,
                   brick_size_t<dim>        brick_sizes,
                   Kokkos::Array<bool, dim> is_brick_periodic);

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  real_t
  get_var(CellLocation_t const & cell_loc) const;

  // ==============================================================
  // ==============================================================
  //! Perform restriction (average value over all sibling)
  KOKKOS_INLINE_FUNCTION
  real_t
  get_var_restriction(CellLocation_t const & cell_loc) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] cellindex_out is the cell index of the ghost cell to fill
   * \param[in] iOct_global is the octant id among all octant owned by current MPI process.
   * \param[in] iOct_out index where to write data
   *
   */
  KOKKOS_INLINE_FUNCTION void
  fill_inner(int32_t cellindex_in,
             int32_t cellindex_out,
             iOct_t  iOct_global,
             iOct_t  iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill a ghost cell data by copying data from a neighbor either at same level, or finer level
   * (doing actually a restriction) a neighbor octant in case neighbor is at the same AMR level.
   *
   * \param[in] cell_loc_out is the cell location where to write data
   * \param[in] cell_loc_in is the cell location where to read data
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_ghost_copy(CellLocation_t const & cell_loc_out,
                  CellLocation_t const & cell_loc_in,
                  index_t const &        cellindex_out,
                  iOct_t const &         iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_neigh,
                                          coord_t<2> const &      coord_in,
                                          iOct_t const &          iOct_global,
                                          index_t const &         cellindex_out,
                                          iOct_t const &          iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_neigh,
                                          coord_t<3> const &      coord_in,
                                          iOct_t const &          iOct_global,
                                          index_t const &         cellindex_out,
                                          iOct_t const &          iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at coarser level.
   *
   * \param[in] key is current octant orchard key
   * \param[in] key_neigh is neighbor octant orchard key
   * \param[in] child_id is the child id of neighbor (at same level) wrt actual neighbor (coarser
   *            level)
   * \param[in] iOct_global index to current octant
   * \param[in] iOct_out index where to write data
   * \param[in] iOct_neigh index to neighbor octant
   * \param[in] cellindex integer used to map the ghost cell to fill
   * \param[in] dir is direction to neighbor (in a 3x3 neighborhood)
   *
   * In 2d, for face neighbors, there are 2 distinct situations :
   *  ______               ______    __
   * |      |             |      |  X  |
   * |      |   __    or  |      |  X__|
   * |      |  X  |       |      |
   * |______|  X__|       |______|
   *
   * In this function, we want to fill the "X" ghost cells using data from
   * the (larger) neighbor octant.
   *
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_ghost_coarser_level(key_t                              key,
                           key_t                              key_neigh,
                           uint8_t                            child_id,
                           int32_t                            iOct_global,
                           iOct_t                             iOct_out,
                           iOct_t                             iOct_neigh,
                           index_t                            cellindex,
                           Kokkos::Array<int8_t, dim> const & dir) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct_global) from
   * a neighbor octant in case neighbor is at finer level.
   *
   * \param[in] key_neigh_same_level is orchard key of neighbor in direction dir at same AMR level
   * \param[in] iOct_global is index to current octant
   * \param[in] iOct_out index where to write data
   * \param[in] cellindex integer used to map the ghost cell to fill
   * \param[in] dir is direction to neighbor (in a 3x3 neighborhood)
   *
   * current  octant (large one) is on the right
   * neighbor octant (small one) is on the left
   *
   * In 2d, for face neighbor, there are 2 distinct situations :
   *        _______             __      _______
   *       |       |           |  |    X       |
   *  __   |       |      or   |__|    X       |
   * |  |  X       |                   |       |
   * |__|  x_______|                   |_______|
   *
   * In 3d, for face neighbor, there are 4 distinct situations :
   */
  KOKKOS_INLINE_FUNCTION
  bool
  fill_ghost_finer_level(key_t                              key_neigh_same_level,
                         iOct_t                             iOct_global,
                         iOct_t                             iOct_out,
                         index_t                            cellindex,
                         Kokkos::Array<int8_t, dim> const & dir) const;

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data all around current octant (iOct_global).
   *
   * This is (almost) the main entry point of the functor, i.e. directly called inside operator().
   *
   * \param[in] cellindex integer used to map the ghost cell to fill
   * \param[in] coord cartesian coordinates of current cell inside block
   * \param[in] iOct_global is index to current octant
   * \param[in] iOct_out is index to where to write data
   *
   */
  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(index_t const &      cellindex,
              coord_t<dim> const & coord,
              iOct_t const &       iOct_global,
              iOct_t const &       iOct_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor when computing in all group quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeAllQuad const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor when computing only mirror quadrant
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeMirrorQuad const &, const index_t & global_index) const;

}; // class ComputeGhostedScalarField

// explicit template instantiation
extern template class ComputeGhostedScalarField<2, kalypsso::DefaultDevice>;
extern template class ComputeGhostedScalarField<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEGHOSTEDSCALARFIELD_H_
