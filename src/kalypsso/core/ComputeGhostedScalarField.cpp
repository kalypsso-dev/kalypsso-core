// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeGhostedScalarField.cpp
 */
#include <kalypsso/core/ComputeGhostedScalarField.h>

namespace kalypsso
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ComputeGhostedScalarField<dim, device_t>::ComputeGhostedScalarField(
  StencilHelper_t              stencil_helper,
  AMRMeshInfo                  amr_mesh_info,
  int32_t                      iOct_begin,
  DataArrayBlock_t             userdata_in,
  DataArrayGhostedBlock_t      userdata_out,
  int32_t                      ivar,
  CellCenteredProlongationType prolongation)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device()
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(iOct_begin)
  , m_userdata_in(userdata_in)
  , m_userdata_out(userdata_out)
  , m_ivar(ivar)
  , m_prolongation(prolongation)

{
  KOKKOS_ASSERT(userdata_in.block_size() == userdata_out.block_size() &&
                "userdata_in and userdata_out must have the same block sizes.");
}

// ==============================================================
// ==============================================================
// same as above, but specifying also the mirror keys array
template <size_t dim, typename device_t>
ComputeGhostedScalarField<dim, device_t>::ComputeGhostedScalarField(
  StencilHelper_t              stencil_helper,
  orchard_key_view_t           mirror_orchard_keys,
  AMRMeshInfo                  amr_mesh_info,
  DataArrayBlock_t             userdata_in,
  DataArrayGhostedBlock_t      userdata_out,
  int32_t                      ivar,
  CellCenteredProlongationType prolongation)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device(mirror_orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(0) // not used when processing mirrors quad
  , m_userdata_in(userdata_in)
  , m_userdata_out(userdata_out)
  , m_ivar(ivar)
  , m_prolongation(prolongation)
{}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeGhostedScalarField<dim, device_t>::apply_on_group(amr_hashmap_t      amr_hashmap,
                                                         orchard_key_view_t orchard_keys,
                                                         AMRMeshInfo        amr_mesh_info,
                                                         int32_t            iOct_begin,
                                                         int32_t            num_octants_in_group,
                                                         DataArrayBlock_t   userdata_in,
                                                         DataArrayGhostedBlock_t  userdata_out,
                                                         int32_t                  ivar,
                                                         brick_size_t<dim>        brick_sizes,
                                                         Kokkos::Array<bool, dim> is_brick_periodic)
{
  // make sure the range of octants to process is valid
  assertm((iOct_begin + num_octants_in_group) <= amr_mesh_info.local_num_quadrants(),
          "Invalid range of octants to process");

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  const auto prolongation_type = CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD;

  ComputeGhostedScalarField<dim, device_t> functor(
    stencil_helper, amr_mesh_info, iOct_begin, userdata_in, userdata_out, ivar, prolongation_type);

  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_octants_in_group * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeGhostedScalarField",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeGhostedScalarField<dim, device_t>::apply_in_mirrors(
  amr_hashmap_t            amr_hashmap,
  orchard_key_view_t       orchard_keys,
  orchard_key_view_t       mirror_orchard_keys,
  AMRMeshInfo              amr_mesh_info,
  DataArrayBlock_t         userdata_in,
  DataArrayGhostedBlock_t  userdata_out,
  int32_t                  ivar,
  brick_size_t<dim>        brick_sizes,
  Kokkos::Array<bool, dim> is_brick_periodic)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  const auto prolongation_type = CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD;

  ComputeGhostedScalarField<dim, device_t> functor(stencil_helper,
                                                   mirror_orchard_keys,
                                                   amr_mesh_info,
                                                   userdata_in,
                                                   userdata_out,
                                                   ivar,
                                                   prolongation_type);

  const auto num_mirrors = mirror_orchard_keys.extent(0);
  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_mirrors * static_cast<size_t>(nbCellsPerGhostedLeaf);

  assertm(num_mirrors == static_cast<size_t>(amr_mesh_info.local_num_mirrors()),
          "wrong number of mirror quads.");

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ComputeGhostedScalarField",
                       Kokkos::RangePolicy<exec_space, TagComputeMirrorQuad>(0, nbCellsTotal),
                       functor);

} // apply_in_mirrors

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ComputeGhostedScalarField<dim, device_t>::get_var(CellLocation_t const & cell_loc) const
{
  const auto   cellindex_in = cell_loc.cellindex(m_userdata_in.block_size());
  const auto & iOct_in = cell_loc.iOct;

  return m_userdata_in(cellindex_in, m_ivar, iOct_in);
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ComputeGhostedScalarField<dim, device_t>::get_var_restriction(CellLocation_t const & cell_loc) const
{

  auto const & block_size = m_userdata_in.block_size();

  return m_stencil_helper.compute_siblings_average(cell_loc, block_size, m_ivar, m_userdata_in);

} // get_var_restriction

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::fill_inner(int32_t cellindex_in,
                                                     int32_t cellindex_out,
                                                     iOct_t  iOct_global,
                                                     iOct_t  iOct_out) const
{
  // read variable
  const auto value = m_userdata_in(cellindex_in, m_ivar, iOct_global);

  // write variable
  m_userdata_out(cellindex_out, 0, iOct_out) = value;

} // fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::fill_ghost_copy(CellLocation_t const & cell_loc_out,
                                                          CellLocation_t const & cell_loc_in,
                                                          index_t const &        cellindex_out,
                                                          iOct_t const &         iOct_out) const
{

  const bool do_restriction = cell_loc_in.level() == (cell_loc_out.level() + 1);

  // read variable
  const auto value = do_restriction ? get_var_restriction(cell_loc_in) : get_var(cell_loc_in);

  // write variable
  m_userdata_out(cellindex_out, 0, iOct_out) = value;

} // fill_ghost_copy

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const &         cell_loc_neigh,
  coord_t<2> const &              coord_in,
  [[maybe_unused]] iOct_t const & iOct_global,
  index_t const &                 cellindex_out,
  iOct_t const &                  iOct_out) const
{
  const real_t slope_type = 1;

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;

  // compute limited slopes
  auto const dudx = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, m_ivar, m_userdata_in, slope_type);
  auto const dudy = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, m_ivar, m_userdata_in, slope_type);

  // extrapolate conservative variables
  real_t value =
    m_userdata_in(
      cell_loc_neigh.cellindex(m_userdata_in.block_size()), m_ivar, cell_loc_neigh.iOct) +
    ONE_FOURTH_F * static_cast<real_t>(ix) * dudx + ONE_FOURTH_F * static_cast<real_t>(iy) * dudy;

  m_userdata_out(cellindex_out, 0, iOct_out) = value;

} // linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const &         cell_loc_neigh,
  coord_t<3> const &              coord_in,
  iOct_t const &                  iOct_global,
  index_t const &                 cellindex_out,
  [[maybe_unused]] iOct_t const & iOct_out) const
{

  const real_t slope_type = 1; // TODO : investigate if a better value should be searched for

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
  const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;
  const int iz = 2 * (coord_in[IZ] - 2 * (coord_in[IZ] / 2)) - 1;

  // compute limited slopes
  auto const dudx = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, m_ivar, m_userdata_in, slope_type);
  auto const dudy = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, m_ivar, m_userdata_in, slope_type);
  auto const dudz = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_z, cell_loc_left_z, m_ivar, m_userdata_in, slope_type);

  // extrapolate conservative variables
  real_t value =
    m_userdata_in(
      cell_loc_neigh.cellindex(m_userdata_in.block_size()), m_ivar, cell_loc_neigh.iOct) +
    ONE_FOURTH_F * static_cast<real_t>(ix) * dudx + ONE_FOURTH_F * static_cast<real_t>(iy) * dudy +
    ONE_FOURTH_F * static_cast<real_t>(iz) * dudz;

  m_userdata_out(cellindex_out, 0, iOct_global - m_iOct_begin) = value;

} // linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
                                                      coord_t<dim> const & coord_out,
                                                      iOct_t const &       iOct_global,
                                                      iOct_t const &       iOct_out) const
{

  const auto & b = m_userdata_in.block_size();

  // coordinates of source cell (where to read data)
  coord_t<dim> coord_in;
  const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

  int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_userdata_in.block_size());

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  if (dir_norm == 0)
  {
    // current cell is inside current block
    fill_inner(cellindex_in, cellindex_out, iOct_global, iOct_out);
  }
  else
  {
    // current cell is a ghost cell (thus belonging to a neighbor block)

    /*
     * fill actual ghosts with data from a neighbor block.
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_global);

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const CellLocation_t cell_loc_cur{ coord_in, key_cur, iOct_global, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     */
    if (cell_loc_neigh.level() >= cell_loc_cur.level())
    {
      // doing a simple copy or doing a restriction when neighbor is at higher AMR level
      fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, iOct_out);
    }
    else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser

      if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value
        fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, iOct_out);
      }
      else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        linear_extrapolate_using_limited_slopes(
          cell_loc_neigh, coord_in, iOct_global, cellindex_out, iOct_out);
      }
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm ==0)

} // fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::operator()(TagComputeAllQuad const &,
                                                     const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve local octant index (this is where we want to write data)
  const auto iOct_local = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct_local * nbCellsPerGhostedLeaf;

  // retrieve global octant index
  const auto iOct_global = m_iOct_begin + iOct_local;

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iOct_local);

} // operator() - TagComputeAllQuad

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeGhostedScalarField<dim, device_t>::operator()(TagComputeMirrorQuad const &,
                                                     const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve mirror index
  const auto iMirror = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iMirror * nbCellsPerGhostedLeaf;

  // retrieve key associated to that mirror index
  const auto mirror_key = m_mirror_orchard_keys_device(iMirror);

  // make sure the key is in the hashmap and retrieve value
  const auto mirror_hashindex = m_stencil_helper.m_amr_hashmap_device.find(mirror_key);
  [[maybe_unused]] const auto valid =
    m_stencil_helper.m_amr_hashmap_device.valid_at(mirror_hashindex);

  KOKKOS_ASSERT(
    valid && "(mirror quadrant) key doesn't exist in hashmap (this is in principle not possible, "
             "since mirror keys are computed from p4est ghosts.)");

  // retrieve iOct associated to that mirror quadrant
  const auto iOct_global = m_stencil_helper.m_amr_hashmap_device.value_at(mirror_hashindex);

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iMirror);

} // operator() - TagComputeMirrorQuad

// explicit template instantiation
template class ComputeGhostedScalarField<2, kalypsso::DefaultDevice>;
template class ComputeGhostedScalarField<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
