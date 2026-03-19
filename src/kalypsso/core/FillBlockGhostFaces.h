// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostFaces.h
 */
#ifndef KALYPSSO_CORE_FILLBLOCKGHOSTFACES_H_
#define KALYPSSO_CORE_FILLBLOCKGHOSTFACES_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h> // for orchard_key_view_t alias
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/prolongation.h>

namespace kalypsso
{

/**
 * \class FillBlockGhostFacesFunctor
 *
 * This is a kokkos functor class that takes as input a FaceDataArrayBlock<dim, device_t> and as
 * output another FaceDataArrayBlock<dim, device_t> (with larger block size), and for each
 * quadrant, fill block ghost faces.
 * Again, just to be clear, the input array is non-ghosted, output array is ghosted.
 *
 * Limitations:
 * - this class only supports block sizes that are even integers.
 * - this class only supports ghost sizes that are even integers.
 *
 * As a reminder:
 * - external faces are faces with even coordinate in the normal direction
 * - internal faces are faces with odd  coordinate in the normal direction
 *
 * \note the main difference with class FillBlockGhostCellsFunctor is that here we do things in two
 * steps (hence the definition of two tagged operator() ).
 *
 * - First step computes all faces except news faces (called internal faces) when doing
 * prolongation.
 * - Second step computes new internal faces in the ghost zones when doing prolongation.
 *
 * Prolongating external faces is done using either:
 * - simple copy
 * - linear extrapolation using limited slopes (when neighbor block is coarser than current block)
 *
 * Prolongation internal faces is done using a divergence preserving algorithm
 * - currently only Toth and Roe (2002) is implemented, reference
 *   Toth and Roe, JCP, 180, 746-759, 2002: Divergence- and curl-preserving
 * prolongation and restriction formulas. https://doi.org/10.1006/jcph.2002.7120
 *
 *
 * - on the left, we have a coarse cell that has 4 faces
 * - on the right, we have 4 fine cells, 8 external faces, and 4 internal faces (labeled with "+")
 *
 *    AMR level l                    AMR level l+1
 * ________________                ________ ________
 * |              |                |       +       |
 * |              |  prolongation  |       +       |
 * |              |     ====>      |       +       |
 * |              |                 +++++++ +++++++
 * |              |     <====      |       +       |
 * |              |  restriction   |       +       |
 * |______________|                |_______+______ |
 *
 *
 * See file test_AMRmesh_fill_block_ghosts_faces.cpp for testing
 *
 * \note Important note: which quadrant must be filled ?
 * - MPI owned quadrants : inner part and ghost cells
 * - MPI ghost quadrants : inner part and ghost cells
 * - outside quadrants : only the inner part which will be used when computing primitive variables
 */
template <size_t dim, typename device_t>
class FillBlockGhostFacesFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using FaceLocation_t = FaceLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  //! Compute everything: fill ghost faces (inner block + external faces in ghost zone)
  struct TagFillAllFacesButInternal
  {};

  //! Finalize filling ghost faces (internal faces in ghost zone)
  struct TagFillInternalFaces
  {};

  /**
   *
   * \param[in] stencil helper
   * \param[in] iOct_begin is the first octant to process
   * \param[in] num_octants is the number of octant to process
   * \param[in] facedata_in non-ghosted face data array
   * \param[in,out] facedata_out ghosted face data array
   * \param[in] prolongation selects how coarse neighbor must be prolongated to fill ghost faces
   */
  FillBlockGhostFacesFunctor(StencilHelper_t      stencil_helper,
                             int32_t              iOct_begin,
                             int32_t              num_octants,
                             FaceDataArrayBlock_t facedata_in,
                             FaceDataArrayBlock_t facedata_out,
                             ProlongationParam    prolongation,
                             AMRMeshInfo          amr_mesh_info)
    : m_stencil_helper(stencil_helper)
    , m_iOct_begin(iOct_begin)
    , m_num_octants(num_octants)
    , m_facedata_in(facedata_in)
    , m_facedata_out(facedata_out)
    , m_block_sizes(facedata_out.cell_block_size_inner())
    , m_total_sizes(facedata_out.cell_block_size())
    , m_prolongation(prolongation)
    , m_amr_mesh_info(amr_mesh_info)
  {}

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        [[maybe_unused]] int32_t local_num_octants,
        int32_t                  iOct_begin,
        int32_t                  num_octants,
        FaceDataArrayBlock_t     facedata_in,
        FaceDataArrayBlock_t     facedata_out,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic,
        AMRMeshInfo              amr_mesh_info);

  // ==============================================================
  // ==============================================================
  /**
   * Given a face multi-index identifying a face in the ghost zone, determine is face is ambiguous,
   * i.e. shared by two neighbor blocks.
   *
   * Faces that are in the ghost zones and aligned with the block border are said "ambiguous"
   * because those faces are colocated with two neighbor blocks (top and below in the following
   * drawing).
   *
   * Two situations may happen:
   * - if top and below blocks are at the same AMR level, no problem, they both agree on the value
   * attached to the ambiguous face.
   * - if "top block" (direct face neighbor) is at level l-1 (i.e. coarser than current block) and
   * "below block" is at level l, we must use face value from "below block" (if not, we may have
   * artefacts in prolongating values from the "top block", that can be seen e.g. when computing
   * divergence).
   *
   *     | ghost zone |
   *   __|____________|__
   *     |            |__
   *     |  Block of  |__
   *     |  cells at  |__                             neighbor block top at level l-1
   *     |  AMR level |__ <== non-ambiguous face      /\
   *     |     l      |__                             ||
   *   __|____________|__ <== ambiguous     face   ___||_________
   *     |            |                               ||
   *     |            |                               \/
   *                                                  neighbor block below at level l
   */
  KOKKOS_INLINE_FUNCTION
  bool
  is_face_ambiguous(face_multiindex_t<dim> const & face_indexes,
                    block_size_t<dim> const &      block_sizes) const;

  // ==============================================================
  // ==============================================================
  /**
   * Convert face indexes out (ghosted) in face indexes in (non ghost)
   *
   * \return relative direction to neighbor block where to look for data to copy
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<int8_t, dim>
  convert_face_indexes_out_to_in(face_multiindex_t<dim> &       face_indexes_in,
                                 face_multiindex_t<dim> const & face_indexes_out) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do a prolongation by linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(FaceLocation<2> const & face_loc_neigh,
                                          FaceLocation<2> const & face_loc_out,
                                          int32_t const &         iOct_local) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do a prolongation by linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(FaceLocation<3> const & face_loc_neigh,
                                          FaceLocation<3> const & face_loc_out,
                                          int32_t const &         iOct_local) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  internal_faces_prolongation_by_toth_and_roe(face_multiindex_t<dim> const & face_indexes_out,
                                              int32_t const &                iOct_local) const;

  // ==============================================================
  // ==============================================================
  /**
   * fill interior of ghosted block.
   *
   * \param[in] cellindex_out is the cell index of the ghost cell to fill
   * \param[in] iOct_global is the octant id among all octant owned by current MPI process.
   *
   * Just to be clear iOct_global - m_iOct_begin is the local octant id inside the group of octant
   * being processed.
   */
  KOKKOS_INLINE_FUNCTION void
  fill_inner(face_multiindex_t<dim> const & face_indexes_in,
             face_multiindex_t<dim> const & face_indexes_out,
             int32_t                        iOct_local) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_all_faces_but_internal(face_multiindex_t<dim> const & face_indexes_out,
                              int32_t const &                iOct_local) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION void
  fill_internal_faces_ghost(face_multiindex_t<dim> const & face_indexes_out,
                            int32_t const &                iOct_local) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFillAllFacesButInternal const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagFillInternalFaces const &, const index_t & global_index) const;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  FaceDataArrayBlock_t m_facedata_in;

  //! a ghosted data array (which block ghost cells need to be filled)
  FaceDataArrayBlock_t m_facedata_out;

  //! block sizes (inner size of ghost block, in unit of cells, not faces)
  const block_size_t<dim> m_block_sizes;

  //! total sizes (ghosted block size, in unit of cells, not faces)
  const block_size_t<dim> m_total_sizes;

  //! prolongation type
  const ProlongationParam m_prolongation;

  //! AMR mesh info (useful for knowing where outside quadrants are)
  const AMRMeshInfo m_amr_mesh_info;

}; // class FillBlockGhostFacesFunctor

extern template class FillBlockGhostFacesFunctor<2, kalypsso::DefaultDevice>;
extern template class FillBlockGhostFacesFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_FILLBLOCKGHOSTFACES_H_
