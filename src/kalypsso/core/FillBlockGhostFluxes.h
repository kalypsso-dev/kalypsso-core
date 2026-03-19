// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostFluxes.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTFLUXES_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTFLUXES_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h> // for orchard_key_view_t alias
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>

namespace kalypsso
{

/**
 * \class FillBlockGhostFluxesFunctor
 *
 * Just consider an AMR mesh where the leaves are populated with block of cells of size (bx,by,bz),
 * and consider an input flux array fluxes_in.
 * If the input corresponds to flux along X direction, then fluxes_in must be a DataArrayBlock of
 * size (bx+1, by, bz).
 *
 * The purpose of this functor is to output a DataArrayGhostedBlock of size
 * (bx+1+2+ghost_width, by, bz), that contains ghost elements that are filled by either
 * - copying data from a neighbor octant at same AMR level
 * - prolongating data from a neighbor octant at coarser AMR level
 * - restricting data from a neighbor octant at finer AMR level
 *
 * Just to clear:
 * - the input array is non-ghosted, output array is ghosted.
 * - this class only supports cell block sizes that are even integers.
 * - the ghost_width can not be strictly larger than bx/2 (resp. by/2 or bz/2); if not we would need
 * to potentially access data from neighbor of neighbor octant which is not allowed since our MPI
 * ghost exchange operator only transfer a ghost region of one octant all around current MPI
 * sub-domain.
 *
 * This class can be used in a piecewise loop over leaf octant.
 *
 * Just to illustrate, let's consider 4x4 blocks of cells, an input flux array along X direction
 * must have shape 5x4. Assume we want to fill 2 ghost flux along the X direction.
 *
 *     Input flux block        Output ghost flux array (2 ghosts on each side)
 *        5x4                      9x4
 *
 *       x x x x x            o o x x x x x o o
 *       x x x x x      ==>   o o x x x x x o o
 *       x x x x x            o o x x x x x o o
 *       x x x x x            o o x x x x x o o
 *
 *
 * The main difficulty here is to deploy the entire combinatorics of
 * geometrical possibilities in terms of
 * - size of neighbor octant, i.e.
 *   is neighbor octant smaller, same size or larger than current octant,
 * - 2d/3d
 *
 * So we need to be careful, have good testing code.
 * See file test_AMRmesh_fill_block_ghosts_flux.cpp
 *
 */
template <size_t dim, typename device_t>
class FillBlockGhostFluxesFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using FaceLocation_t = FaceLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  /**
   *
   * \param[in] stencil helper
   * \param[in] iOct_begin is the first octant to process
   * \param[in] num_octants is the number of octant to process
   * \param[in] userdata_in data array used to fill ghost of userdata_out
   * \param[in,out] userdata_out data array which we want to fill the block ghosts
   *                cells
   * \param[in] prolongation selects how coarse neighbor must be prolongated to fill ghost cells
   */
  FillBlockGhostFluxesFunctor(StencilHelper_t         stencil_helper,
                              int32_t                 iOct_begin,
                              int32_t                 num_octants,
                              ComponentIndex3D        direction,
                              block_size_t<dim>       cell_block_size,
                              DataArrayBlock_t        fluxes_in,
                              DataArrayGhostedBlock_t fluxes_out,
                              int                     var_index_in,
                              int                     var_index_out)
    : m_stencil_helper(stencil_helper)
    , m_iOct_begin(iOct_begin)
    , m_num_octants(num_octants)
    , m_direction(direction)
    , m_cell_block_size(cell_block_size)
    , m_fluxes_in(fluxes_in)
    , m_fluxes_out(fluxes_out)
    , m_var_index_in(var_index_in)
    , m_var_index_out(var_index_out)
  {}

  // ====================================================================
  // ====================================================================
  static void
  check_args_validity(DataArrayBlock_t const &        fluxes_in,
                      DataArrayGhostedBlock_t const & fluxes_out,
                      block_size_t<dim> const &       cell_block_size,
                      ComponentIndex3D                direction);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(amr_hashmap_t             amr_hashmap,
        orchard_key_view_t        orchard_keys,
        [[maybe_unused]] int32_t  local_num_octants,
        int32_t                   iOct_begin,
        int32_t                   num_octants,
        block_size_t<dim> const & cell_block_size,
        ComponentIndex3D          direction,
        DataArrayBlock_t          fluxes_in,
        DataArrayGhostedBlock_t   fluxes_out,
        int                       var_index_in,
        int                       var_index_out,
        brick_size_t<dim>         brick_sizes,
        Kokkos::Array<bool, dim>  is_brick_periodic);

  // ==============================================================
  // ==============================================================
  //! compute flat index inside octant
  KOKKOS_INLINE_FUNCTION int32_t
  to_flat_index(face_multiindex_t<dim> const & face_multiindex,
                block_size_t<dim> const &      flux_shapes) const;

  // ==============================================================
  // ==============================================================
  //! compute face multi-index inside octant
  KOKKOS_INLINE_FUNCTION face_multiindex_t<dim>
  to_face_multiindex(int32_t const & flat_index, block_size_t<dim> const & flux_shapes) const;

  // ==============================================================
  // ==============================================================
  //! compute face multi-index inside octant
  KOKKOS_INLINE_FUNCTION face_multiindex_t<dim>
                         to_face_multiindex(int32_t const &           flat_index,
                                            block_size_t<dim> const & flux_shapes,
                                            shift_t<dim> const &      shift) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] flat_index_in is the cell index of the cell to read data from
   * \param[in] flat_index_out is the cell index of the ghost cell to fill
   * \param[in] iOct_global is the octant id among all octant owned by current MPI process.
   *
   * Just to be clear iOct_global - m_iOct_begin is the local octant id inside the group of octant
   * being processed.
   */
  KOKKOS_INLINE_FUNCTION
  void
  fill_inner(int32_t flat_index_in, int32_t flat_index_out, int32_t iOct_global) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(index_t const &                flat_index_out,
              face_multiindex_t<dim> const & facecoord_out,
              int32_t const &                iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index) const;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! direction associated to flux
  const ComponentIndex3D m_direction;

  //! cell block size (bx,by,bz)
  const block_size_t<dim> m_cell_block_size;

  //! a flux data array (no ghosts, sizes must cell block size plus one in a given direction)
  DataArrayBlock_t m_fluxes_in;

  //! a ghosted data array (which block ghost cells need to be filled)
  DataArrayGhostedBlock_t m_fluxes_out;

  //! variable index in input array
  const int m_var_index_in;

  //! variable index in output array
  const int m_var_index_out;

}; // class FillBlockGhostFluxesFunctor

extern template class FillBlockGhostFluxesFunctor<2, kalypsso::DefaultDevice>;
extern template class FillBlockGhostFluxesFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTFLUXES_H_
