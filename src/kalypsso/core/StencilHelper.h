// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StencilHelper.h
 */
#ifndef KALYPSSO_CORE_STENCILHELPER_H_
#define KALYPSSO_CORE_STENCILHELPER_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/EdgeDataArrayBlock_utils.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/Locations.h>
#include <kalypsso/core/Kokkos_extensions.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex

#include <kalypsso/utils/log/kalypsso_log.h>

namespace kalypsso
{

// ====================================================================================
// ====================================================================================
// ====================================================================================
class EdgeUtils
{
private:
  using edge_shift_t = Kokkos::Array<int, 9>;

  KOKKOS_FUNCTION static int
  edge_shift_x(int i)
  {
    static constexpr edge_shift_t EDGE_SHIFT_X{ -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    return EDGE_SHIFT_X[i];
  }

  KOKKOS_FUNCTION static int
  edge_shift_y(int i)
  {
    const edge_shift_t EDGE_SHIFT_Y{ -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    return EDGE_SHIFT_Y[i];
  }

  KOKKOS_FUNCTION static int
  index_to_shift(int i)
  {
    // clang-format off
    const Kokkos::Array<int, 12> INDEX_TO_SHIFT{ 0, 1, 3,    /* neighbors of EDGE_00 */
                                                 1, 2, 5,    /* neighbors of EDGE_01 */
                                                 3, 6, 7,    /* neighbors of EDGE_10 */
                                                 5, 7, 8 };  /* neighbors of EDGE_11 */
    // clang-format on
    return INDEX_TO_SHIFT[i];
  }

public:
  // =========================================================================================
  // =========================================================================================
  /**
   * For a given edge, return shift to one of the 3 other cells that share this common edge.
   *
   * In 2d, an edge is always along Z axis (so its also a corner).
   *
   * An edge is identified by enum CellEdgeLocation.
   *
   * In the drawing below, we represent the example situation where edge is EDGE_00 (i.e. lower left
   * corner), then the three shifts vector are
   * (-1, -1), (0,-1) and (-1,0) given in that order so that the cells pointed to are in Morton
   * order.
   *               ______
   *              |     |
   *              |     |
   *  (-1,0) ---  X_____|
   *            / |
   *           /  |
   *          /   |
   *  (-1,-1)    (0,-1)
   *
   * \param[in] cell_edge_location
   * \param[in] nth_shift integer indicating which shift user wants (must be O, 1, or 2; there are
   * only possibilities)
   *
   */
  template <size_t dim>
  KOKKOS_FUNCTION static shift_t<dim>
  edge_neighbor_shift(CellEdgeLocation cell_edge_location, int nth_shift, ComponentIndex3D edge_dir)
  {

    if constexpr (dim == 2)
    {
      KOKKOS_ASSERT((nth_shift >= 0 and nth_shift <= 2) &&
                    "nth_shift wrong value must be 0, 1 or 2.");

      const auto index = index_to_shift(3 * cell_edge_location + nth_shift);

      return shift_t<2>{ edge_shift_x(index), edge_shift_y(index) };
    }
    else if constexpr (dim == 3)
    {
      KOKKOS_ASSERT((nth_shift >= 0 and nth_shift <= 2) &&
                    "nth_shift wrong value must be 0, 1 or 2.");

      const auto index = index_to_shift(3 * cell_edge_location + nth_shift);

      if (edge_dir == IZ)
      {
        return shift_t<3>{ edge_shift_x(index), edge_shift_y(index), 0 };
      }
      else if (edge_dir == IX)
      {
        return shift_t<3>{ 0, edge_shift_x(index), edge_shift_y(index) };
      }
      else if (edge_dir == IY)
      {
        return shift_t<3>{ edge_shift_x(index), 0, edge_shift_y(index) };
      }

      return shift_t<3>{ 0, 0, 0 };
    }

  } // edge_neighbor_shift

}; // class EdgeUtils

// =========================================================================================
// =========================================================================================
/**
 * For a given edge (in 3d) along edge_dir, return shift to one of the 3 other cells that share
 * this common edge.
 */
template <size_t dim>
KOKKOS_FUNCTION CellEdgeLocation
get_CellEdgeLocation(EdgeLocation<dim> const & edge_loc, block_size_t<dim> const & block_size);

// ====================================================================================
// ====================================================================================
// ====================================================================================
/**
 * \class StencilHelper
 *
 * \tparam dim dimension
 * \tparam device_t Kokkos device
 *
 * This a helper class for doing stencil operation without using block ghost, i.e. we provide direct
 * access to neighbor values.
 *
 * We also provide default behavior for accessing cells outside domain.
 */
template <size_t dim, typename device_t>
class StencilHelper
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using CellLocation_t = CellLocation<dim>;
  using FaceLocation_t = FaceLocation<dim>;
  using EdgeLocation_t = EdgeLocation<dim>;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, real_t, device_t>;

  // ====================================================================
  // ====================================================================
  //! constructor.
  StencilHelper(amr_hashmap_t            amr_hashmap,
                orchard_key_view_t       orchard_keys,
                block_size_t<dim>        block_sizes,
                brick_size_t<dim>        brick_sizes,
                Kokkos::Array<bool, dim> is_brick_periodic);

  virtual ~StencilHelper() = default;

  KOKKOS_INLINE_FUNCTION
  auto
  key(iOct_t iOct) const
  {
    return m_orchard_keys_device(iOct);
  }

  // =========================================================================================
  // =========================================================================================
  /**
   * Return unit shift.
   *
   * -XDIR, +XDIR, -YDIR, +YDIR, -ZDIR, +ZDIR
   */
  KOKKOS_INLINE_FUNCTION
  auto
  unit_shift(int dir) const
  {
    KOKKOS_ASSERT((dir >= -static_cast<int>(dim) and dir <= static_cast<int>(dim) and dir != 0) &&
                  "WRONG value");

    shift_t<dim> shift;

    for (int i = 1; i <= static_cast<int>(dim); ++i)
    {
      if (dir == i)
        shift[i - 1] = 1;
      else if (dir == -i)
        shift[i - 1] = -1;
      else
        shift[i - 1] = 0;
    }
    return shift;
  }

  // =========================================================================================
  // =========================================================================================
  KOKKOS_FUNCTION
  bool
  is_cell_location_at_domain_border(CellLocation_t const & cell_loc) const;

  // =========================================================================================
  // =========================================================================================
  KOKKOS_FUNCTION
  bool
  is_face_location_at_block_border(FaceLocation_t const & face_loc) const;

  // =========================================================================================
  // =========================================================================================
  KOKKOS_FUNCTION bool
  is_edge_location_at_block_border(EdgeLocation_t const & edge_loc) const;

  // =========================================================================================
  // =========================================================================================
  KOKKOS_FUNCTION bool
  is_edge_location_at_domain_border(EdgeLocation_t const & edge_loc) const;

  // =========================================================================================
  // =========================================================================================
  KOKKOS_INLINE_FUNCTION
  bool
  is_brick_periodic(int dir) const
  {
    KOKKOS_ASSERT((dir >= 0 and dir < static_cast<int>(dim)) &&
                  "[StencilHelper::is_brick_periodic] wrong value for dir");
    return m_is_brick_periodic[dir];
  }

  // =========================================================================================
  // =========================================================================================
  /**
   * Compute cell location of the neighbor of a given cell (from a given block)
   *
   * \param[in] cell_loc current cell location
   * \param[in] shift is array of index shift in unit of current block cells (signed values)
   *
   * Note:
   * - if shift points to a cell outside current block, we need to explore mesh to find neighbor
   * octant first, then cell indexes inside the neighbor block.
   *   -- if neighbor octant is at same level, no problem, only one key and one cell to return
   *   -- if neighbor octant is coarser, we return the corresponding coarse cell coord
   *   -- if neighbor octant is finer, we return the cell coord of the first child (also called
   *      eldest)
   * - if shift points to a cell outside domain (and p4est brick connectivity is non-periodic) then
   *   return the periodic cell location inside domain and set is_outside_domain to true
   */
  KOKKOS_FUNCTION
  auto
  getNeighLoc(CellLocation_t const & cell_loc, shift_t<dim> shift) const -> CellLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Compute face location of the neighbor of a given face (from a given block)
   *
   * \param[in] face_loc current face location
   * \param[in] shift is array of index shift in unit of current block cells (signed values)
   *
   * Note:
   * - if shift points to a face outside current block, we need to explore mesh to find neighbor
   * octant first, then face indexes inside the neighbor block.
   *   -- if neighbor octant is at same level, no problem, only one key and one face to return
   *   -- if neighbor octant is coarser, we return the corresponding coarse face coord
   *   -- if neighbor octant is finer, we return the face coord of the first child (also called
   *      eldest)
   * - if shift points to a face outside domain (and p4est brick connectivity is non-periodic) then
   *   return the periodic face location inside domain and set is_outside_domain to true
   */
  KOKKOS_FUNCTION
  auto
  getNeighLoc(FaceLocation_t const & face_loc, shift_t<dim> shift) const -> FaceLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Same as getNeighLoc but here the caller is sure that neighbor is coarser.
   *
   * Besides, this function is intended to be called from block border cells with a shift vector
   * that leads to a coarser block. The caller of this function must have checked before hand that
   * the neighbor is actually coarser (this information is available by looking at conformal status
   * array, which itself is a member of MeshMap).
   *
   * Below is a 4x4 block of cells where outer border cells are flagged with a "x".
   *
   *   ____________
   *  |x |x |x |x |
   *  |__|__|__|__|
   *  |x |  |  |x |
   *  |__|__|__|__|
   *  |x |  |  |x |
   *  |__|__|__|__|
   *  |x |x |x | x|
   *  |__|__|__|__|
   *
   *
   * \param[in] cell_loc current cell location
   * \param[in] shift is array of index shift in unit of current block cells (signed values)
   *
   */
  KOKKOS_FUNCTION
  auto
  getNeighLocCoarser(CellLocation_t const & cell_loc, shift_t<dim> shift) const -> CellLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Same as getNeighLoc but here the caller is sure that neighbor is finer.
   *
   * Besides, this function is intended to be called from block border cells with a shift vector
   * that leads to a finer block. The caller of this function must have checked before hand that
   * the neighbor is actually finer (this information is available by looking at conformal status
   * array, which itself is a member of MeshMap).
   *
   * \param[in] cell_loc current cell location
   * \param[in] shift is array of index shift in unit of current block cells (signed values)
   *
   * \return a cell location to the eldest siblings in the correspond finer octant (in a local group
   * a 2^dim cells).
   */
  KOKKOS_FUNCTION
  auto
  getNeighLocFiner(CellLocation_t const & cell_loc, shift_t<dim> shift) const -> CellLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Same as getNeighLocFiner but we modify the returned cell loc to be nearer the original
   * location.
   *
   * Given the location returned by getNeighLocFiner, we shift back the location in the opposite
   * shift direction by one cell.
   *
   * \param[in] cell_loc current cell location
   * \param[in] shift is array of index shift in unit of current block cells (signed values)
   *
   */
  KOKKOS_FUNCTION
  auto
  getNeighLocFinerNearer(CellLocation_t const & cell_loc,
                         shift_t<dim>           shift) const -> CellLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Given a face location (corresponding to a block border), compute symmetric face location as
   * seen from the other side.
   *
   * Some example situation using 4x4 block of cells.
   *
   * Example 1:
   * - face location on the left  block is is the input face location
   * - face location on the right block is is the output face location
   *
   *     ________             ________
   *    |_|_|_|_|            |_|_|_|_|
   *    |_|_|_|_|  in   out  |_|_|_|_|
   *    |_|_|_|_| <--- --->  |_|_|_|_|
   *    |_|_|_|_|            |_|_|_|_|
   *
   * \note if input face location is strictly inside block (not at block border), we return itself
   *
   * \param[in] face_loc current face location
   * \return symmetric face location
   */
  KOKKOS_FUNCTION
  auto
  getBorderFaceLocSymmetric(FaceLocation_t const & face_loc) const -> FaceLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   */
  KOKKOS_FUNCTION
  auto
  edge_to_cell_location(EdgeLocation_t const & edge_loc) const -> CellLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Given a edge location (at block border), compute symmetric edge location as
   * seen from the other side assuming the other is either at same or finer AMR level; in all other
   * cases, we return the input edge location as default invalid value.
   *
   * An example situation using 4x4 block of cells.
   * Remember that in 2d, only edges along Z have a meaning, i.e. edge are cell corner in 2d.
   *
   * Example 1 in 2d:
   * - edge location "x" on the left block is is the input face location
   * - edge location on the right block is is the output face location
   *
   *
   * Input block             ________
   *  ________________      |_|_|_|_|
   * |   |   |   |   |      |_|_|_|_|
   * |___|___|___|___|      |_|_|_|_|
   * |   |   |   |   |      #_|_|_|_|
   * |___|___|___|___#       _______
   * |   |   |   |   |      |_|_|_|_|
   * |___|___|___|___x  =>  x_|_|_|_|
   * |   |   |   |   |      |_|_|_|_|
   * |___|___|___|___o      |_|_|_|_|
   *
   * - if input edge is also a block edge, e.g. at corner labeled with "o", there are 3 possible
   * answers corresponding to the 3 other blocks touching "o". To choose which one is desired, we
   * use enum EdgeNormalType. When there is only one possibility, this parameter is unused.
   * - if input edge is exactly in the middle of a non-conformal face (e.g. node labeled with "#"),
   * there are actually 2 possibles symmetric corresponding nodes: the upper-left corner of bottom
   * right block, or the lower-left corner of upper block (also labeled with "#"), identified as the
   * block with the highest octant id (greater in the Morton order sense).
   *
   * \param[in] edge_loc current edge location
   * \param[in] edge_normal_type
   * \return symmetric edge location
   */
  KOKKOS_FUNCTION
  auto
  getBorderEdgeLocSymmetric(EdgeLocation_t const & edge_loc,
                            EdgeNormalType edge_normal_type = EdgeNormalType::DIAGONAL) const
    -> EdgeLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Given an input edge location return one of the sibling edge location (sharing the same edge).
   *
   * In 2D, an edge is always shared by at most four siblings edge locations (see drawing below).
   * If input edge location is at AMR level l, then the sibling edge locations (noted x0, x1 and x2)
   * either live at level l, l+1 or l,l-1, but it can not be a mix between l-1, l and l+1 (because
   * of 2:1 balance constraint). As we are looking for one of the 3 sibling edge locations, we must
   * provide in input index 0, 1 or 2 (which enumerate the sibling in Morton order, see function
   * EdgeUtils::edge_neighbor_shift).
   *
   * When an edge is a on a T (hanging face), with no corresponding siblings on the other side,
   * there are only 2 siblings. In that case we return an edge location for which field is_valid is
   * false.
   *
   * In 3D, there is a special case, if one of the neighbor cell is at finer AMR level (i.e. l+1),
   * there are actually two possible edge locations that can be returned, here we always return
   * the edge location corresponding to the lowest Morton index.
   *
   * Idea of the algorithm applied here:
   *
   * - step 1. convert the input edge location in to a pair (cell_location, cell_edge_location)
   *    note that: for all edge inside a block, we consider the edge belongs to the cell on its
   * right, except the last edge on the block which belongs to the cell on its left (N cells / N+1
   * edges)
   * - step 2. apply the cell shift
   * - step 3. apply the edge mirroring and return the new edge location
   *
   *   ______________________
   *  |          |          |
   *  |          |          |
   *  |          |          |
   *  |          |          |
   *  |        x | x2       |
   *  |__________|__________|
   *  |       x0 | x1  |    |
   *  |          |     |    |
   *  |          |_____|____|
   *  |          |     |    |
   *  |          |     |    |
   *  |__________|_____|____|
   *
   * \param[in] edge_loc_in input edge location (labeled "x")
   * \param[in] nth_shift integer (0, 1 or 2) which identifies one of the 3 neighbor edge siblings
   * location.
   *
   */
  KOKKOS_FUNCTION
  auto
  getEdgeSiblingLoc(EdgeLocation_t const & edge_loc_in, int nth_shift) const -> EdgeLocation_t;

  // =========================================================================================
  // =========================================================================================
  /**
   * Given an edge location, this edge is shared by 4 cells, return the 3 other edge locations
   * corresponding to the same edge but seen from the neighbor cells.
   *
   * Example situation: let's consider an edge shared by four cells, suppose in input, edge location
   * labeled by "x" is known, and we want in output the 3 edge locations labeled "x0", "x1" and
   * "x2", that determine in which octant they live, at which AMR level, ... the full geometrical
   * information.
   *
   *   ______________________
   *  |          |          |
   *  |          |          |
   *  |          |          |
   *  |          |          |
   *  |        x | x2       |
   *  |__________|__________|
   *  |       x0 | x1  |    |
   *  |          |     |    |
   *  |          |_____|____|
   *  |          |     |    |
   *  |          |     |    |
   *  |__________|_____|____|
   *
   * \note if the four cells sharing the input edge are in the same octant, obviously all edge
   * locations will be identical. This routine is only really useful when at one of the edge
   * location belong to different octant.
   *
   * \param[in] edge_loc current edge location
   * \param[out] edge_loc0 current edge location
   * \param[out] edge_loc1 current edge location
   * \param[out] edge_loc2 current edge location
   */
  KOKKOS_FUNCTION
  void
  getAllEdgeSiblingLoc(EdgeLocation_t const & edge_loc,
                       EdgeLocation_t &       edge_loc0,
                       EdgeLocation_t &       edge_loc1,
                       EdgeLocation_t &       edge_loc2) const;

  // ==============================================================
  // ==============================================================
  //! compute average of all sibling cells given a cell location.
  //!
  //! the cell location must have a xyz field containing even integers; this is actually the case
  //! when the cell location is returned by getNeighLoc for a neighbor at finer level.
  //!
  //! For example, given a 8x8 block grid, and given cell marked with "o", all the siblings are
  //! marked with "x". The lower left sibling is called the eldest sibling.
  //!    _______________
  //!   |_|_|_|_|_|_|_|_|
  //!   |_|_|_|_|_|_|_|_|
  //!   |_|_|_|_|_|_|_|_|
  //!   |_|_|_|_|_|_|_|_|
  //!   |_|_|x|o|_|_|_|_|
  //!   |_|_|x|x|_|_|_|_|
  //!   |_|_|_|_|_|_|_|_|
  //!   |_|_|_|_|_|_|_|_|
  //!
  //!
  //! \param[in] cell_loc is current cell
  //! \param[in] bloc_ksizes
  //! \param[in] ivar identify which variable we want to average
  //! \param[in] userdata
  //!
  //! \return average value
  KOKKOS_FUNCTION real_t
  compute_siblings_average(CellLocation_t const &    cell_loc,
                           block_size_t<dim> const & block_sizes,
                           int                       ivar,
                           DataArrayBlock_t const &  userdata) const;

  // ==============================================================
  // ==============================================================
  //! This does the same thing as 'compute_siblings_average' when input is a
  //! 'DataArrayGhostedBlock'.
  //!
  //! \param[in] cell_loc is current cell
  //! \param[in] bloc_ksizes
  //! \param[in] ivar identify which variable we want to average
  //! \param[in] userdata
  //!
  //! \return average value
  KOKKOS_FUNCTION real_t
  compute_siblings_average(CellLocation_t const &          cell_loc,
                           int                             ivar,
                           DataArrayGhostedBlock_t const & userdata) const;

  // ==============================================================
  // ==============================================================
  //! This does the same thing as 'compute_siblings_average' with a 'DataArrayBlockMultiVar'
  //! instead.
  //!
  //! \param[in] cell_loc is current cell
  //! \param[in] bloc_ksizes
  //! \param[in] ivar identify which variable we want to average
  //! \param[in] userdata
  //!
  //! \return average value
  KOKKOS_FUNCTION real_t
  compute_siblings_average(CellLocation_t const &           cell_loc,
                           block_size_t<dim> const &        block_sizes,
                           int                              ivar,
                           DataArrayBlockMultiVar_t const & userdata) const;

  // ==============================================================
  // ==============================================================
  //!
  //! compute face siblings average in a group of cell-sibling (same parent).
  //!
  //! In other words, given the coordinate of a given cell, let's consider its 2^dim neighborhood
  //! (associated to virtual parent coarse cell). We perform the average of a given face field of
  //! over all face siblings of this cell.
  //!   ____________
  //!  |     |     |
  //!  |  o  |     |
  //!  |_____|_____|
  //!  |     |     |
  //!  | x o |  x  |
  //!  |_____|_____|
  //!
  //! \note:
  //! - the lower left cell marked with "x" and "o" is the eldest cell of all siblings in a block of
  //!   2^dim cells.
  //! - the two cells marked with "x" are face-siblings for face = YMIN
  //! - the two cells marked with "o" are face-siblings for face = XMIN
  //!
  //! There are 2*dim distinct faces, hence 2*dim averages possible : XMIN/XMAX/YMIN/YMAX/ZMIN/ZMAX
  //!
  //! The implementation is cell-wise oriented because it is easier the reason about a block 2^dim
  //! cells, rather than faces.
  //!
  //! \note We can only average X-component over X-siblings, Y-component over Y-siblings, ...
  //!
  //! Just to be clear, when, e.g.
  //! - ivar=IX and ivar_face=LEFT, we are averaging the left x-face of the 2 cells marked with "o"
  //! - ivar=IX and ivar_face=RIGHT, we are averaging the right x-face of the 2 cells marked with
  //! "o" (i.e. the internal vertical faces of the 2^dim block of cells).
  //! - ivar=IY and ivar_face=LEFT, we are averaging By over the left face of the 2 cells marked
  //! with "x"
  //!
  //! \param[in] cell_loc contains the CELL location of one sibling.
  //! \param[in] ivar variable id : IX, IY or IZ (even in 2d; magnetic field can have a Z component
  //! in 2d)
  //! \param[in] ivar_face : LEFT or RIGHT
  //! \param[in] Bface a FaceDataArrayBlock_t container (e.g. magnetic field)
  //!
  //! \return average value
  //!
  KOKKOS_FUNCTION real_t
  compute_face_siblings_average(CellLocation_t const &       cell_loc,
                                int                          ivar,
                                face_type_t                  ivar_face,
                                FaceDataArrayBlock_t const & Bface) const;

  // ==============================================================
  // ==============================================================
  //!
  //! Compute average of face variables on the left side (same parent).
  //!
  //! In other words, given the coordinate of a given cell, let's consider its 2^dim neighborhood
  //! (associated to virtual parent coarse cell). We perform the average of a given face field of
  //! over all face siblings of this cell.
  //!   ____________
  //!  |     |     |
  //!  o     |     |
  //!  |_____|_____|
  //!  |     |     |
  //!  o     |     |
  //!  |__x__|__x__|
  //!
  //!
  //! \param[in] face_loc contains the FACE location of the eldest sibling.
  //! \param[in] cell_block_size array of number of cells along each direction.
  //! \param[in] ivar is the variable identifying what should be averaged.
  //! \param[in] data is a data array (assume to be a flux array).
  //!
  //! \note Remember that a flux array is an array which block sizes are one unit larger in one
  //! direction that the cell block size.
  //! As an example, consider a cell block of size (bx,by,bz). A flux array along Y direction must
  //! be of size (bx,by+1,bz).
  KOKKOS_FUNCTION real_t
  compute_face_siblings_average(FaceLocation_t const &    face_loc,
                                block_size_t<dim> const & cell_block_size,
                                int32_t                   ivar,
                                DataArrayBlock_t const &  data) const;

  // ==============================================================
  // ==============================================================
  //!
  //! compute face siblings average in a group of cell-sibling (same parent coarse cell).
  //!
  //! Here, the input cell location MUST be the eldest child (as given by the output of getNeighLoc
  //! in case neighbor is finer); in other words the cell coordinates must be all even integers.
  //! We perform the average of a given face field
  //! all face siblings.
  //!
  //!      ^       ^
  //!   ___|_______|____
  //!  |       |       |
  //!  |       |       |
  //!  |->     |       |->      horizontal arrow : Bx
  //!  |       |       |        vertical   arrow : By
  //!  |_______|_______|
  //!  |       |       |
  //!  |       |       |
  //!  |-> E   |       |->
  //!  |   ^   |   ^   |
  //!  |___|___|___|___|
  //!
  //! \note:
  //! - the lower left cell marked with "E" is the eldest child of all siblings.
  //! - we draw the face-centered arrow for the external faces
  //!
  //! There are 2*dim distinct faces, hence 2*dim averages possible : XMIN/XMAX/YMIN/YMAX/ZMIN/ZMAX
  //!
  //! \note We can only average X-component over X-siblings, Y-component over Y-siblings, ...
  //!
  //!
  //! \param[in] cell_loc contains the CELL location of the eldest sibling (must be all even
  //! integers).
  //! \param[in] ivar variable id : IX, IY or IZ (even in 2d; magnetic field can have a Z component
  //! in 2d)
  //! \param[in] ivar_face : LEFT or RIGHT external face
  //! \param[in] Bface a FaceDataArrayBlock_t container (e.g. magnetic field)
  //!
  //! \return average value
  //!
  KOKKOS_FUNCTION real_t
  compute_external_face_siblings_average(CellLocation_t const &       cell_loc,
                                         int                          ivar,
                                         face_type_t                  ivar_face,
                                         FaceDataArrayBlock_t const & Bface) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_FUNCTION real_t
  compute_face_siblings_average(FaceLocation_t const &       cell_loc,
                                FaceDataArrayBlock_t const & Bface) const;


  // ==============================================================
  // ==============================================================
  //! compute sum over face neighbor values in a 2^dim cell mesh.
  //!
  //! Example situation:
  //! - consider 4x4 block of cells,
  //! - consider a cell located at i=3,j=2 (labeled with symbol "c"),
  //!
  //! One wants to sum data in "c" with its 2^(dim-1) face siblings; it 2d it amounts to suming "c"
  //! with either data from "c1" on the left or from "c2" above.
  //!
  //! This function becomes interesting when "c" is a cell that touches the outer block limit, and
  //! if the neighbor block is at a different AMR level (e.g. coarser level)
  //!
  //!    ___________________________________________________________________________ ...
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |        |   c2   |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |________|________|________|________|        n        |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |   c1   |   c    |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |________|________|________|________|_________________|_________________|___ ...
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |________|________|________|________|                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |        |        |        |        |                 |                 |
  //!   |________|________|________|________|_________________|_________________|____ ...
  //!
  //!
  //! when the block on the right is at coarser level, we may need to update cell "n" by suming
  //! fluxes data coming from cell "c" and "c2.". This is useful in a finite volume scheme, one
  //! needs to sum all the fluxes at a non-conform interface.
  //!
  //! Code is written in a generic way to identify face-siblings in a 2^dim neighborhood.
  //!
  //! There are 2 interfaces to this function:
  //!
  //! \param[in] cell_loc (can be obtained by calling getNeighLoc, beware in case the output cell is
  //! in a finer block, cell_loc will be populated with ijk coordinates of the eldest child of the
  //! local 2^dim cell neighborhood).
  //!
  //! \param[in] ivar identifies which variable we want to average
  //!
  //! \param[in] userdata a DataArrayBlock (assumes same shapes as block_sizes, except in one
  //!            direction, where size is one more block_sizes)
  //!
  //! \param[in] direction is used to identify the face siblings participating to
  //!            the sum (it should be coherent with the shift used to find the neighbor octant)
  //!            In other words, direction can be IX, IY or IZ, it identifies a direction orthogonal
  //!            to the shift.
  //!
  //! \note important note, cell block sizes are deduced from userdata shape. Don't use this
  //! interface when suming flux.
  //!
  //! \return sum value
  KOKKOS_FUNCTION real_t
  compute_face_siblings_sum(CellLocation_t const &   cell_loc,
                            int                      ivar,
                            DataArrayBlock_t const & userdata,
                            int                      dir) const;


  // ==============================================================
  // ==============================================================
  //! compute sum over face neighbor values in a DataArrayBlock that is a flux array in some
  //! direction.
  KOKKOS_FUNCTION real_t
  compute_face_siblings_sum(CellLocation_t const &   cell_loc,
                            int                      ivar,
                            DataArrayBlock_t const & userdata,
                            bool                     use_right_flux) const;

  // ==============================================================
  // ==============================================================
  //! evaluate minmod slopes with neighbor at finer or same AMR level.
  //!
  //! \param[in] cell_loc_cur   current location
  //! \param[in] cell_loc_right right   location
  //! \param[in] cell_loc_left  left    location
  //! \param[in] ivar variable index
  //! \param[in] userdata a DataArrayBlock used for computing slopes
  //! \param[in] slope_type
  //!
  //! important note:
  //! let l be the AMR level at current location.
  //! We assume left and right neighbors locations are at level l or l+1 but NOT l-1 (with would
  //! imply a prolongation).
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes(CellLocation_t const &   cell_loc_cur,
                        CellLocation_t const &   cell_loc_right,
                        CellLocation_t const &   cell_loc_left,
                        int                      ivar,
                        DataArrayBlock_t const & userdata,
                        real_t                   slope_type) const;

  // ==============================================================
  // ==============================================================
  //! evaluate minmod slopes with neighbor at finer or same AMR level.
  //!
  //! \param[in] cell_loc_cur   current location
  //! \param[in] cell_loc_right right   location
  //! \param[in] cell_loc_left  left    location
  //! \param[in] ivar variable index
  //! \param[in] userdata a DataArrayBlock used for computing slopes
  //! \param[in] slope_type
  //!
  //! important note:
  //! let l be the AMR level at current location.
  //! We assume left and right neighbors locations are at level l or l+1 but NOT l-1 (with would
  //! imply a prolongation).
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes(CellLocation_t const &          cell_loc_cur,
                        CellLocation_t const &          cell_loc_right,
                        CellLocation_t const &          cell_loc_left,
                        int                             ivar,
                        DataArrayGhostedBlock_t const & userdata,
                        real_t                          slope_type) const;

  // ==============================================================
  // ==============================================================
  //! Equivalent of 'compute_minmod_slopes' for 'DataArrayBlockMultiVar'
  //!
  //! \param[in] cell_loc_cur   current location
  //! \param[in] ivar_cur variable index of current
  //! \param[in] cell_loc_right right   location
  //! \param[in] ivar_right variable index of right
  //! \param[in] cell_loc_left  left    location
  //! \param[in] ivar_left variable index of left
  //! \param[in] userdata a DataArrayBlockMultiVar used for computing slopes
  //! \param[in] slope_type
  //!
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes(CellLocation_t const &           cell_loc_cur,
                        int                              ivar_cur,
                        CellLocation_t const &           cell_loc_right,
                        int                              ivar_right,
                        CellLocation_t const &           cell_loc_left,
                        int                              ivar_left,
                        DataArrayBlockMultiVar_t const & userdata,
                        real_t                           slope_type) const;

  // ==============================================================
  // ==============================================================
  //! evaluate minmod slopes without any level constraint.
  //!
  //! This variant is to be used when doing prolongation from a coarse level to fine level.
  //!
  //! \param[in] cell_loc_cur   current location
  //! \param[in] cell_loc_right right   location
  //! \param[in] cell_loc_left  left    location
  //! \param[in] ivar variable index
  //! \param[in] userdata a DataArrayBlock used for computing slopes
  //! \param[in] slope_type
  //!
  //! important note:
  //! let l be the AMR level at current location.
  //! We assume left and right neighbors locations maybe at any level l-1,l,l+1. When neighbor is at
  //! level l-1, it would it in principle require a prolongation, but then one would have a chicken
  //! and egg problem. So it that case, we do a simple copy.
  //!
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes_prolongation(CellLocation_t const &   cell_loc_cur,
                                     CellLocation_t const &   cell_loc_right,
                                     CellLocation_t const &   cell_loc_left,
                                     int                      ivar,
                                     DataArrayBlock_t const & userdata,
                                     real_t                   slope_type) const;

  // ==============================================================
  // ==============================================================
  //! Equivalent of 'compute_minmod_slopes_prolongation' for 'DataArrayBlockMultiVar'
  //!
  //! \param[in] cell_loc_cur   current location
  //! \param[in] ivar_cur variable index of current
  //! \param[in] cell_loc_right right   location
  //! \param[in] ivar_right variable index of right
  //! \param[in] cell_loc_left  left    location
  //! \param[in] ivar_left variable index of left
  //! \param[in] userdata a DataArrayBlockMultiVar used for computing slopes
  //! \param[in] slope_type
  //!
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes_prolongation(CellLocation_t const &           cell_loc_cur,
                                     int                              ivar_cur,
                                     CellLocation_t const &           cell_loc_right,
                                     int                              ivar_right,
                                     CellLocation_t const &           cell_loc_left,
                                     int                              ivar_left,
                                     DataArrayBlockMultiVar_t const & userdata,
                                     real_t                           slope_type) const;


  // ==============================================================
  // ==============================================================
  //! evaluate minmod slopes with neighbor at finer or same AMR level.
  //!
  //! \param[in] face_loc_cur   current location
  //! \param[in] face_loc_right right   location
  //! \param[in] face_loc_left  left    location
  //! \param[in] userdata a FaceDataArrayBlock used for computing slopes
  //!
  //! important note:
  //! let l be the AMR level at current location.
  //! We assume left and right neighbors locations maybe at any level l-1,l,l+1. When neighbor is at
  //! level l-1, it would it in principle require a prolongation, but then one would have a chicken
  //! and egg problem. So it that case, we do a simple copy.
  //!
  KOKKOS_FUNCTION real_t
  compute_minmod_slopes_prolongation(FaceLocation_t const &       face_loc_cur,
                                     FaceLocation_t const &       face_loc_right,
                                     FaceLocation_t const &       face_loc_left,
                                     FaceDataArrayBlock_t const & facedata) const;

  // ==============================================================
  // ==============================================================
  //! evaluate linear combination with neighbors at finer or same AMR level.
  //!
  //! \param[in] cell_loc_left  left    location
  //! \param[in] cell_loc_cur   current location
  //! \param[in] cell_loc_right right   location
  //! \param[in] target_level AMR level at which data must be used
  //! \param[in] ivar variable index
  //! \param[in] userdata a DataArrayBlock
  //! \param[in] coef
  //!
  //! important note:
  //!
  //! We assume that all cell locations are at level target_level or target_level+1 but NOT
  //! target_level-1 (with would imply a prolongation).
  KOKKOS_FUNCTION real_t
  compute_linear_combination(CellLocation_t const &         cell_loc_left,
                             CellLocation_t const &         cell_loc_cur,
                             CellLocation_t const &         cell_loc_right,
                             uint8_t                        target_level,
                             int const                      ivar,
                             DataArrayBlock_t const &       userdata,
                             Kokkos::Array<real_t, 3> const coef) const;

  // ==============================================================
  // ==============================================================
  //! evaluate linear combination with neighbors at finer or same AMR level.
  //!
  //! \param[in] cell_loc_left  left2    location
  //! \param[in] cell_loc_left  left    location
  //! \param[in] cell_loc_cur   current location
  //! \param[in] cell_loc_right right   location
  //! \param[in] cell_loc_right right2   location
  //! \param[in] target_level AMR level at which data must be used
  //! \param[in] ivar variable index
  //! \param[in] userdata a DataArrayBlock
  //! \param[in] coef
  //!
  //! important note:
  //!
  //! We assume that all cell locations are at level target_level or target_level+1 but NOT
  //! target_level-1 (with would imply a prolongation).
  KOKKOS_FUNCTION real_t
  compute_linear_combination(CellLocation_t const &         cell_loc_left2,
                             CellLocation_t const &         cell_loc_left,
                             CellLocation_t const &         cell_loc_cur,
                             CellLocation_t const &         cell_loc_right,
                             CellLocation_t const &         cell_loc_right2,
                             uint8_t                        target_level,
                             int const                      ivar,
                             DataArrayBlock_t const &       userdata,
                             Kokkos::Array<real_t, 5> const coef) const;


public:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

}; // class StencilHelper

// explicit template instantiation
extern template class StencilHelper<2, kalypsso::DefaultDevice>;
extern template class StencilHelper<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class StencilHelper<2, kalypsso::HostDevice>;
extern template class StencilHelper<3, kalypsso::HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_STENCILHELPER_H
