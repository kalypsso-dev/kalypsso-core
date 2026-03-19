// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MultiMatFillBlockGhostCells.h
 */
#ifndef KALYPSSO_CORE_MULTI_MAT_FILL_BLOCK_GHOST_CELLS_H_
#define KALYPSSO_CORE_MULTI_MAT_FILL_BLOCK_GHOST_CELLS_H_

#include <kalypsso/core/FillBlockGhosts_common.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/MaterialPresence.h>

namespace kalypsso
{

// ================================================================================================
// ================================================================================================

/**
 * \class MultiMatFillBlockGhostCellsFunctor
 *
 * Does the same thing as FillBlockGhostCellsFunctor using a DataArrayBlockMultiVar and material
 * presence array.
 *
 * TODO: Would it be interesting to make a common class with FillBlockGhostCellsFunctor?
 * TODO: Precompute material number for destination in advance?
 */
template <size_t dim, typename device_t>
class MultiMatFillBlockGhostCellsFunctor
{
public:
  using AmrHashmap_t = typename hashmap_base_t<device_t>::map_t;
  using OrchardKeys_t = typename orchard_key_base_t<device_t>::view_t;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;
  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, real_t, device_t>;
  using DataArrayGhostedBlockMultiVar_t = DataArrayGhostedBlockMultiVar<dim, real_t, device_t>;
  using StencilHelper_t = StencilHelper<dim, device_t>;
  using CellLocation_t = CellLocation<dim>;

  static void
  apply(ConfigMap const &               config_map,
        AmrHashmap_t                    amr_hashmap,
        OrchardKeys_t                   orchard_keys,
        int32_t                         start_octant,
        int32_t                         end_octant,
        DataArrayBlockMultiVar_t        userdata_in,
        MaterialPresenceView_t          in_mat,
        DataArrayGhostedBlockMultiVar_t userdata_out,
        MaterialPresenceView_t          out_mat,
        int32_t                         num_vars_per_mat,
        Kokkos::Array<bool, dim>        is_brick_periodic)
  {
    auto stencil_helper = StencilHelper_t(amr_hashmap,
                                          orchard_keys,
                                          userdata_in.block_size(),
                                          get_brick_sizes<dim>(config_map),
                                          is_brick_periodic);

    MultiMatFillBlockGhostCellsFunctor<dim, device_t> functor(
      stencil_helper,
      userdata_in,
      in_mat,
      userdata_out,
      out_mat,
      get_cell_prolongation_type(config_map),
      num_vars_per_mat);

    const auto num_cells = Kokkos::dim_prod(userdata_out.shape());
    Kokkos::RangePolicy<typename device_t::execution_space> policy(start_octant * num_cells,
                                                                   end_octant * num_cells);

    // for AMR tree leaf, explore the neighbor block
    Kokkos::parallel_for("kalypsso::MultiMatFillBlockGhostCellsFunctor", policy, functor);
  }

  KOKKOS_INLINE_FUNCTION void
  operator()(const int32_t global_index) const
  {
    const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

    const auto iOct_global = global_index / nbCellsPerGhostedLeaf;
    const auto cell_index = global_index - iOct_global * nbCellsPerGhostedLeaf;

    const auto coord_out =
      cellindex_to_coord<dim>(cell_index, m_userdata_out.shape(), m_userdata_out.shift());

    fill_ghosts(cell_index, coord_out, iOct_global);
  }

private:
  // ==============================================================================================
  // ==============================================================================================
  // INNER FUNCTIONS ADAPTED FROM FillBlockGhostCellsFunctor
  // ==============================================================================================
  // ==============================================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_neigh,
                                          coord_t<2> const &      coord_in,
                                          int32_t const &         cellindex_out,
                                          int32_t const &         iOct_global) const
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

    // determine local position of current cell inside virtual parent cell using integer coordinates
    // in -1, +1
    const int ix = 2 * (coord_in[IX] - 2 * (coord_in[IX] / 2)) - 1;
    const int iy = 2 * (coord_in[IY] - 2 * (coord_in[IY] / 2)) - 1;

    const auto nbmat = m_out_mat.num_materials(iOct_global);

    for (int32_t imat = 0; imat < nbmat; ++imat)
    {
      // Extrapolation needs variable at each face so we check if it is present across each face.
      const auto mat_num = m_out_mat.material_num(iOct_global, imat);
      const bool has_mat = m_in_mat.get(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_neigh.iOct), mat_num);

      if (has_mat) // Is the material present in the source octant?
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto in_ivar_neigh =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_neigh.iOct), mat_num);

          const auto in_ivar_left_x =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num);

          const auto in_ivar_right_x =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num);

          const auto in_ivar_left_y =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num);

          const auto in_ivar_right_y =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num);

          const auto out_ivar = ivar + m_num_vars_per_mat * imat;

          // compute limited slopes
          auto const dudx = m_stencil_helper.compute_minmod_slopes(cell_loc_neigh,
                                                                   in_ivar_neigh,
                                                                   cell_loc_right_x,
                                                                   in_ivar_right_x,
                                                                   cell_loc_left_x,
                                                                   in_ivar_left_x,
                                                                   m_userdata_in,
                                                                   slope_type);

          auto const dudy = m_stencil_helper.compute_minmod_slopes(cell_loc_neigh,
                                                                   in_ivar_neigh,
                                                                   cell_loc_right_y,
                                                                   in_ivar_right_y,
                                                                   cell_loc_left_y,
                                                                   in_ivar_left_y,
                                                                   m_userdata_in,
                                                                   slope_type);

          // extrapolate
          m_userdata_out(cellindex_out, out_ivar, iOct_global) =
            m_userdata_in(
              cell_loc_neigh.cellindex(m_block_sizes), in_ivar_neigh, cell_loc_neigh.iOct) +
            KALYPSSO_NUM(0.25) * static_cast<real_t>(ix) * dudx +
            KALYPSSO_NUM(0.25) * static_cast<real_t>(iy) * dudy;
        }
      else // If not, set the value to 0 (maybe best to switch extrapolation method?)
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto out_ivar = ivar + m_num_vars_per_mat * imat;
          m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
        }
    }
  }

  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_neigh,
                                          coord_t<3> const &      coord_in,
                                          int32_t const &         cellindex_out,
                                          int32_t const &         iOct_global) const
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

    const auto nbmat = m_out_mat.num_materials(iOct_global);

    for (int32_t imat = 0; imat < nbmat; ++imat)
    {
      // Extrapolation needs variable at each face so we check if it is present across each face.
      const auto mat_num = m_out_mat.material_num(iOct_global, imat);
      const bool has_mat = m_in_mat.get(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_left_z.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_right_z.iOct), mat_num) &&
                           m_in_mat.get(static_cast<int32_t>(cell_loc_neigh.iOct), mat_num);

      if (has_mat) // Is the material present in the source octant?
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto in_ivar_neigh =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_neigh.iOct), mat_num);

          const auto in_ivar_left_x =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_left_x.iOct), mat_num);

          const auto in_ivar_right_x =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_right_x.iOct), mat_num);

          const auto in_ivar_left_y =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_left_y.iOct), mat_num);

          const auto in_ivar_right_y =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_right_y.iOct), mat_num);

          const auto in_ivar_left_z =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_left_z.iOct), mat_num);

          const auto in_ivar_right_z =
            ivar + m_num_vars_per_mat *
                     m_in_mat.material_index(static_cast<int32_t>(cell_loc_right_z.iOct), mat_num);

          const auto out_ivar = ivar + m_num_vars_per_mat * imat;

          // compute limited slopes
          auto const dudx = m_stencil_helper.compute_minmod_slopes(cell_loc_neigh,
                                                                   in_ivar_neigh,
                                                                   cell_loc_right_x,
                                                                   in_ivar_right_x,
                                                                   cell_loc_left_x,
                                                                   in_ivar_left_x,
                                                                   m_userdata_in,
                                                                   slope_type);

          auto const dudy = m_stencil_helper.compute_minmod_slopes(cell_loc_neigh,
                                                                   in_ivar_neigh,
                                                                   cell_loc_right_y,
                                                                   in_ivar_right_y,
                                                                   cell_loc_left_y,
                                                                   in_ivar_left_y,
                                                                   m_userdata_in,
                                                                   slope_type);

          auto const dudz = m_stencil_helper.compute_minmod_slopes(cell_loc_neigh,
                                                                   in_ivar_neigh,
                                                                   cell_loc_right_z,
                                                                   in_ivar_right_z,
                                                                   cell_loc_left_z,
                                                                   in_ivar_left_z,
                                                                   m_userdata_in,
                                                                   slope_type);

          // extrapolate
          m_userdata_out(cellindex_out, out_ivar, iOct_global) =
            m_userdata_in(
              cell_loc_neigh.cellindex(m_block_sizes), in_ivar_neigh, cell_loc_neigh.iOct) +
            KALYPSSO_NUM(0.25) * static_cast<real_t>(ix) * dudx +
            KALYPSSO_NUM(0.25) * static_cast<real_t>(iy) * dudy +
            KALYPSSO_NUM(0.25) * static_cast<real_t>(iz) * dudz;
        }
      else // If not, set the value to 0 (maybe best to switch extrapolation method?)
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto out_ivar = ivar + m_num_vars_per_mat * imat;
          m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
        }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void
  fill_inner(int32_t cellindex_in, int32_t cellindex_out, int32_t iOct_global) const
  {
    const auto nbmat = m_out_mat.num_materials(iOct_global);

    for (int32_t imat = 0; imat < nbmat; ++imat)
    {
      const auto mat_num = m_out_mat.material_num(iOct_global, imat);

      if (m_in_mat.get(iOct_global, mat_num)) // Is the material present in the source octant?
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto in_ivar =
            ivar + m_num_vars_per_mat * m_in_mat.material_index(iOct_global, mat_num);
          const auto out_ivar = ivar + m_num_vars_per_mat * imat;

          m_userdata_out(cellindex_out, out_ivar, iOct_global) =
            m_userdata_in(cellindex_in, in_ivar, iOct_global);
        }
      else
        for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
        {
          const auto out_ivar = ivar + m_num_vars_per_mat * imat;
          m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
        }
    }
  }

  KOKKOS_INLINE_FUNCTION void
  fill_ghosts(int32_t const &      cellindex_out,
              coord_t<dim> const & coord_out,
              int32_t const &      iOct_global) const
  {

    const auto & b = m_block_sizes;

    coord_t<dim> coord_in;
    const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

    int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_block_sizes);

    auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
    if constexpr (dim == 3)
      dir_norm += dir[IZ] * dir[IZ];

    if (dir_norm == 0)
    {
      // current cell is inside current block
      fill_inner(cellindex_in, cellindex_out, iOct_global);
    }
    else
    {
      // current cell is a ghost cell (thus belonging to a neighbor block)

      /*
       * fill ghosts all around
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
      const auto iOct_in = static_cast<int32_t>(cell_loc_neigh.iOct);
      const auto nbmat = m_out_mat.num_materials(iOct_global);

      if (cell_loc_neigh.level() == cell_loc_cur.level())
      {
        // doing a simple copy
        cellindex_in = cell_loc_neigh.cellindex(m_block_sizes);

        for (int32_t imat = 0; imat < nbmat; ++imat)
        {
          const auto mat_num = m_out_mat.material_num(iOct_global, imat);

          if (m_in_mat.get(iOct_in, mat_num)) // Is the material present in the source octant?
            for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
            {
              const auto in_ivar =
                ivar + m_num_vars_per_mat * m_in_mat.material_index(iOct_in, mat_num);
              const auto out_ivar = ivar + m_num_vars_per_mat * imat;

              m_userdata_out(cellindex_out, out_ivar, iOct_global) =
                m_userdata_in(cellindex_in, in_ivar, iOct_in);
            }
          else
            for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
            {
              const auto out_ivar = ivar + m_num_vars_per_mat * imat;
              m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
            }
        }
      }
      else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
      {
        // doing a prolongation because neighbor is coarser

        if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
        {
          // simple copy of the coarse value

          cellindex_in = cell_loc_neigh.cellindex(m_block_sizes);

          for (int32_t imat = 0; imat < nbmat; ++imat)
          {
            const auto mat_num = m_out_mat.material_num(iOct_global, imat);

            if (m_in_mat.get(iOct_in, mat_num)) // Is the material present in the source octant?
              for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
              {
                const auto in_ivar =
                  ivar + m_num_vars_per_mat * m_in_mat.material_index(iOct_in, mat_num);
                const auto out_ivar = ivar + m_num_vars_per_mat * imat;

                m_userdata_out(cellindex_out, out_ivar, iOct_global) =
                  m_userdata_in(cellindex_in, in_ivar, iOct_in);
              }
            else
              for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
              {
                const auto out_ivar = ivar + m_num_vars_per_mat * imat;
                m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
              }
          }
        }
        else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
        {
          linear_extrapolate_using_limited_slopes(
            cell_loc_neigh, coord_in, cellindex_out, iOct_global);
        }
      }
      else if (cell_loc_neigh.level() == cell_loc_cur.level() + 1)
      {
        for (int32_t imat = 0; imat < nbmat; ++imat)
        {
          const auto mat_num = m_out_mat.material_num(iOct_global, imat);

          if (m_in_mat.get(iOct_in, mat_num)) // Is the material present in the source octant?
            for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
            {
              const auto out_ivar = ivar + m_num_vars_per_mat * imat;
              const auto in_ivar =
                ivar + m_num_vars_per_mat * m_in_mat.material_index(iOct_in, mat_num);

              m_userdata_out(cellindex_out, out_ivar, iOct_global) =
                m_stencil_helper.compute_siblings_average(
                  cell_loc_neigh, m_block_sizes, in_ivar, m_userdata_in);
            }
          else
            for (int32_t ivar = 0; ivar < m_num_vars_per_mat; ++ivar)
            {
              const auto out_ivar = ivar + m_num_vars_per_mat * imat;
              m_userdata_out(cellindex_out, out_ivar, iOct_global) = 0;
            }
        }
      }
      else
      {
        KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
      }

    } // end if (dir_norm == 0)
  }

  // ==============================================================================================
  // ==============================================================================================
  // ==============================================================================================
  // ==============================================================================================
  // ==============================================================================================

  MultiMatFillBlockGhostCellsFunctor(StencilHelper_t                 stencil_helper,
                                     DataArrayBlockMultiVar_t        userdata_in,
                                     MaterialPresenceView_t          in_mat,
                                     DataArrayGhostedBlockMultiVar_t userdata_out,
                                     MaterialPresenceView_t          out_mat,
                                     CellCenteredProlongationType    prolongation,
                                     int32_t                         num_vars_per_mat)
    : m_stencil_helper(stencil_helper)
    , m_in_mat(in_mat)
    , m_out_mat(out_mat)
    , m_userdata_in(userdata_in)
    , m_userdata_out(userdata_out)
    , m_block_sizes(userdata_in.shape())
    , m_num_vars_per_mat(num_vars_per_mat)
    , m_prolongation(prolongation)
  {}

  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! Source material presence
  MaterialPresenceView_t m_in_mat;

  //! Destination material presence
  MaterialPresenceView_t m_out_mat;

  //! a block data array (no ghosts)
  DataArrayBlockMultiVar_t m_userdata_in;

  //! a block data array (ghosts)
  DataArrayGhostedBlockMultiVar_t m_userdata_out;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! num vars per mat
  const int32_t m_num_vars_per_mat;

  //! prolongation type
  const CellCenteredProlongationType m_prolongation;
};

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================

// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================
// ==============================================================================================

/**
 * \class MultiMatFillBlockGhostCellsMatPresence
 *
 * Generates the material presence array of the destination. Used to reorganize the destination
 * matrix
 */
template <size_t dim, typename device_t>
class MultiMatFillBlockGhostCellsMatPresence
{
public:
  using AmrHashmap_t = typename hashmap_base_t<device_t>::map_t;
  using OrchardKeys_t = typename orchard_key_base_t<device_t>::view_t;
  using MaterialPresenceView_t = MaterialPresenceView<device_t>;

  static void
  apply(const MaterialPresenceView_t   src_mat,
        const AmrHashmap_t             hashmap,
        const OrchardKeys_t            keys,
        const brick_size_t<dim>        brick_size,
        const Kokkos::Array<bool, dim> brick_periodicity,
        const int32_t                  start_octant,
        const int32_t                  end_octant,
        MaterialPresenceView_t         dst_mat)
  {
    MultiMatFillBlockGhostCellsMatPresence<dim, device_t> functor(
      src_mat, hashmap, keys, brick_size, brick_periodicity, dst_mat);
    Kokkos::RangePolicy<typename device_t::execution_space> policy(start_octant, end_octant);
    Kokkos::parallel_for("kalypsso::MultiMatFillBlockGhostCellsMatPresence", policy, functor);
  }

  KOKKOS_FUNCTION void
  operator()(const int32_t i_oct) const
  {
    const auto key = m_keys(i_oct);
    MaterialPresenceView_t::copy(m_dst_mat, i_oct, m_src_mat, i_oct);

    // Loop over faces
    for (uint8_t face = 0; face < Face::num_faces<dim>(); face++)
      get_mat_over_face(key, face, i_oct);

    // Loop over edges (if and only if dim == 3)
    if constexpr (dim == 3)
      for (uint8_t edge = 0; edge < Edge::num_edges<dim>(); edge++)
        get_mat_over_edge(key, edge, i_oct);

    // Loop over corners
    for (uint8_t corner = 0; corner < Corner::num_corners<dim>(); corner++)
      get_mat_over_corner(key, corner, i_oct);
  }

private:
  MultiMatFillBlockGhostCellsMatPresence(const MaterialPresenceView_t   src_mat,
                                         const AmrHashmap_t             hashmap,
                                         const OrchardKeys_t            keys,
                                         const brick_size_t<dim>        brick_size,
                                         const Kokkos::Array<bool, dim> brick_periodicity,
                                         const MaterialPresenceView_t   dst_mat)
    : m_src_mat(src_mat)
    , m_dst_mat(dst_mat)
    , m_hashmap(hashmap)
    , m_keys(keys)
    , m_brick_size(brick_size)
    , m_brick_periodicity(brick_periodicity)
  {}

  //! Updates the material presence with the ones over the faces
  KOKKOS_FUNCTION void
  get_mat_over_face(const key_t key, const uint8_t face, const int32_t i_dst_oct) const
  {
    int32_t i_src_oct;

    coord_t<dim, int8_t> dir{};
    dir[face >> 1] = (face & 1) ? 1 : -1;

    const auto neighbor_key =
      orchard_key_t<dim>::get_neighbor_key_same_level(key, dir, m_brick_size, m_brick_periodicity);
    auto neighbor_key_hash = m_hashmap.find(neighbor_key);
    auto is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at same level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_key_coarser = orchard_key_t<dim>::father(neighbor_key);
    neighbor_key_hash = m_hashmap.find(neighbor_key_coarser);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at coarser level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_keys_finer = compute_face_neighbor_key_finer<dim>(neighbor_key, face);
    neighbor_key_hash = m_hashmap.find(neighbor_keys_finer[0]);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at finer level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
      for (uint8_t i = 1; i < neighbor_keys_finer.size(); i++)
      {
        i_src_oct =
          static_cast<int32_t>(m_hashmap.value_at(m_hashmap.find(neighbor_keys_finer[i])));
        MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
      }
    }
  }

  //! Updates the material presence with the ones over the edges (dim 3 only)
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_FUNCTION void
  get_mat_over_edge(const key_t key, const uint8_t edge, const int32_t i_dst_oct) const
  {
    coord_t<3, int8_t> dir{};

    // Taken from 'edge_to_faces'
    if (edge < 4) // edge along Z
    {
      dir[IX] = edge & 0x1 ? 1 : -1;
      dir[IY] = edge & 0x2 ? 1 : -1;
    }
    else if (edge < 8) // edge along X
    {
      dir[IY] = edge & 0x1 ? 1 : -1;
      dir[IZ] = edge & 0x2 ? 1 : -1;
    }
    else // edge along Y (not a circular permutation - respect Morton order)
    {
      dir[IX] = edge & 0x1 ? 1 : -1;
      dir[IZ] = edge & 0x2 ? 1 : -1;
    }

    const auto neighbor_key =
      orchard_key_t<3>::get_neighbor_key_same_level(key, dir, m_brick_size, m_brick_periodicity);
    auto neighbor_key_hash = m_hashmap.find(neighbor_key);
    auto is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    int32_t i_src_oct;

    // Neighbor is at same level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_key_coarser = orchard_key_t<3>::father(neighbor_key);
    neighbor_key_hash = m_hashmap.find(neighbor_key_coarser);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at coarser level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_keys_finer = compute_edge_neighbor_finer_key(neighbor_key, edge);
    neighbor_key_hash = m_hashmap.find(neighbor_keys_finer[0]);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at finer level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
      for (uint8_t i = 1; i < neighbor_keys_finer.size(); i++)
      {
        i_src_oct =
          static_cast<int32_t>(m_hashmap.value_at(m_hashmap.find(neighbor_keys_finer[i])));
        MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
      }
    }
  }

  //! Updates the material presence with the ones over the corners
  KOKKOS_FUNCTION void
  get_mat_over_corner(const key_t key, const uint8_t corner, const int32_t i_dst_oct) const
  {
    coord_t<dim, int8_t> dir{};
    dir[IX] = (corner >> 0 & 1) ? 1 : -1;
    dir[IY] = (corner >> 1 & 1) ? 1 : -1;
    if constexpr (dim == 3)
      dir[IZ] = (corner >> 2 & 1) ? 1 : -1;

    const auto neighbor_key =
      orchard_key_t<dim>::get_neighbor_key_same_level(key, dir, m_brick_size, m_brick_periodicity);
    auto neighbor_key_hash = m_hashmap.find(neighbor_key);
    auto is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    int32_t i_src_oct;

    // Neighbor is at same level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_key_coarser = orchard_key_t<dim>::father(neighbor_key);
    neighbor_key_hash = m_hashmap.find(neighbor_key_coarser);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at coarser level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }

    const auto neighbor_key_finer =
      compute_corner_neighbor_finer_key<dim>(neighbor_key, corner_to_faces<dim>(corner));
    neighbor_key_hash = m_hashmap.find(neighbor_key_finer);
    is_key_valid = m_hashmap.valid_at(neighbor_key_hash);

    // Neighbor is at finer level
    if (is_key_valid)
    {
      i_src_oct = static_cast<int32_t>(m_hashmap.value_at(neighbor_key_hash));
      return MaterialPresenceView_t::update(m_dst_mat, i_dst_oct, m_src_mat, i_src_oct);
    }
  }

  //! Source material presence
  MaterialPresenceView_t m_src_mat;

  //! Destination material presence
  MaterialPresenceView_t m_dst_mat;

  //! keys to index hashmap
  AmrHashmap_t m_hashmap;

  //! index to keys array
  OrchardKeys_t m_keys;

  //! Brick size
  brick_size_t<dim> m_brick_size;

  //! Brick periodicity
  Kokkos::Array<bool, dim> m_brick_periodicity;
};

} // namespace kalypsso

#endif // KALYPSSO_CORE_MULTI_MAT_FILL_BLOCK_GHOST_CELLS_H_
