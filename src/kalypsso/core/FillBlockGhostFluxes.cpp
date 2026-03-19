// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillBlockGhostFluxes.cpp
 * \brief \copybrief FillBlockGhostFluxes.h
 */
#include <kalypsso/core/FillBlockGhostFluxes.h>

namespace kalypsso
{
// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
FillBlockGhostFluxesFunctor<dim, device_t>::check_args_validity(
  DataArrayBlock_t const &        fluxes_in,
  DataArrayGhostedBlock_t const & fluxes_out,
  block_size_t<dim> const &       cell_block_size,
  ComponentIndex3D                direction)
{
  // check that input is a flux array in given direction
  if (fluxes_in.shape()[direction] != (cell_block_size[direction] + 1))
  {
    KALYPSSO_ERROR("fluxes_in shape={} and  cell_bloc_size={}",
                   fluxes_in.shape()[direction],
                   cell_block_size[direction]);
    Kokkos::abort("Input has invalid shape");
  }

  // check fluxes_out shape
  if (fluxes_out.shift()[IX] + cell_block_size[IX] / 2 < 0)
  {
    Kokkos::abort("shift[IX] is too large (negative)");
  }
  if (fluxes_out.shift()[IY] + cell_block_size[IY] / 2 < 0)
  {
    Kokkos::abort("shift[IY] is too large (negative)");
  }

  if constexpr (dim == 3)
  {
    if (fluxes_out.shift()[IZ] + cell_block_size[IZ] / 2 < 0)
    {
      Kokkos::abort("shift[IZ] is too large (negative)");
    }
  }


  if ((fluxes_out.shift()[IX] + fluxes_out.shape()[IX] - fluxes_out.block_size()[IX]) >
      cell_block_size[IX] / 2)
  {
    Kokkos::abort("shift[IX] is too large (negative)");
  }
  if ((fluxes_out.shift()[IY] + fluxes_out.shape()[IY] - fluxes_out.block_size()[IY]) >
      cell_block_size[IY] / 2)
  {
    Kokkos::abort("shift[IY] is too large (negative)");
  }
  if constexpr (dim == 3)
  {
    if ((fluxes_out.shift()[IZ] + fluxes_out.shape()[IZ] - fluxes_out.block_size()[IZ]) >
        cell_block_size[IZ] / 2)
    {
      Kokkos::abort("shift[IZ] is too large (negative)");
    }
  }

} // FillBlockGhostFluxesFunctor<dim, device_t>::check_args_validity

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillBlockGhostFluxesFunctor<dim, device_t>::apply(amr_hashmap_t             amr_hashmap,
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
                                                  Kokkos::Array<bool, dim>  is_brick_periodic)
{

  // make sure the range of octants to process is valid
  KOKKOS_ASSERT(iOct_begin + num_octants <= local_num_octants &&
                "Invalid range of octants to process");

  auto stencil_helper =
    StencilHelper_t(amr_hashmap, orchard_keys, cell_block_size, brick_sizes, is_brick_periodic);

  check_args_validity(fluxes_in, fluxes_out, cell_block_size, direction);

  FillBlockGhostFluxesFunctor<dim, device_t> functor(stencil_helper,
                                                     iOct_begin,
                                                     num_octants,
                                                     direction,
                                                     cell_block_size,
                                                     fluxes_in,
                                                     fluxes_out,
                                                     var_index_in,
                                                     var_index_out);

  const auto nbFluxesPerGhostedLeaf = fluxes_out.num_cells();
  const auto nbFluxesTotal = num_octants * nbFluxesPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for(
    "FillBlockGhostFluxesFunctor", Kokkos::RangePolicy<exec_space>(0, nbFluxesTotal), functor);

} // FillBlockGhostFluxesFunctor<dim, device_t>::apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION int32_t
FillBlockGhostFluxesFunctor<dim, device_t>::to_flat_index(
  face_multiindex_t<dim> const & face_multiindex,
  block_size_t<dim> const &      flux_shapes) const
{
  int32_t index = 0;

  if constexpr (dim == 2)
  {
    return face_multiindex[IX] + flux_shapes[IX] * face_multiindex[IY];
  }
  else if constexpr (dim == 3)
  {
    return face_multiindex[IX] + flux_shapes[IX] * face_multiindex[IY] +
           flux_shapes[IX] * flux_shapes[IY] * face_multiindex[IZ];
  }

  return index;

} // FillBlockGhostFluxesFunctor<dim, device_t>::to_flat_index

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION face_multiindex_t<dim>
                       FillBlockGhostFluxesFunctor<dim, device_t>::to_face_multiindex(
  int32_t const &           flat_index,
  block_size_t<dim> const & flux_shapes) const
{
  KOKKOS_ASSERT(flat_index >= 0);

  face_multiindex_t<dim> res;

  if constexpr (dim == 2)
  {
    const auto & bx = flux_shapes[IX];
    res[IY] = (flat_index / bx);
    res[IX] = (flat_index - bx * res[IY]);
  }
  else if constexpr (dim == 3)
  {
    const auto & bx = flux_shapes[IX];
    const auto & by = flux_shapes[IY];

    res[IZ] = (flat_index / (bx * by));
    int32_t flat_index2 = flat_index - bx * by * res[IZ];
    res[IY] = (flat_index2 / bx);
    res[IX] = (flat_index2 - bx * res[IY]);
  }

  res[dim] = m_direction;

  return res;

} // FillBlockGhostFluxesFunctor<dim, device_t>::to_face_multiindex

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION face_multiindex_t<dim>
                       FillBlockGhostFluxesFunctor<dim, device_t>::to_face_multiindex(
  int32_t const &           flat_index,
  block_size_t<dim> const & flux_shapes,
  shift_t<dim> const &      shift) const
{
  auto res = this->to_face_multiindex(flat_index, flux_shapes);

  res[IX] += shift[IX];
  res[IY] += shift[IY];
  if constexpr (dim == 3)
  {
    res[IZ] += shift[IZ];
  }

  return res;

} // FillBlockGhostFluxesFunctor<dim, device_t>::to_face_multiindex

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFluxesFunctor<dim, device_t>::fill_inner(int32_t flat_index_in,
                                                       int32_t flat_index_out,
                                                       int32_t iOct_global) const
{

  m_fluxes_out(flat_index_out, m_var_index_out, iOct_global - m_iOct_begin) =
    m_fluxes_in(flat_index_in, m_var_index_in, iOct_global);

} // FillBlockGhostFluxesFunctor<dim, device_t>::fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFluxesFunctor<dim, device_t>::fill_ghosts(
  index_t const &                flat_index_out,
  face_multiindex_t<dim> const & face_multiindex_out,
  int32_t const &                iOct_global) const
{

  auto face_multiindex_in = face_multiindex_out;

  // direction to neighbor octant where we need to fetch data to fill ghost flux
  // dir = 0 means we stay in current octant
  int dir = 0;
  if (face_multiindex_out[m_direction] < 0)
  {
    dir = -1;
    face_multiindex_in[m_direction] += m_cell_block_size[m_direction];
  }
  else if (face_multiindex_out[m_direction] > m_cell_block_size[m_direction])
  {
    dir = 1;
    face_multiindex_in[m_direction] -= m_cell_block_size[m_direction];
  }

  //
  // there is a strong assumption here:
  // we assume the flux that are on the block edge have already been corrected when non-conformal
  // interface is detected (this is the case in lagrange-remap scheme, where the star variables have
  // been corrected in the Lagrange phase.)
  //
  if (dir == 0)
  {
    // flux is located inside current block => just copy
    const auto flat_index_in = this->to_flat_index(face_multiindex_in, m_fluxes_in.shape());

    fill_inner(flat_index_in, flat_index_out, iOct_global);
  }
  else
  {
    // flux is ghost (thus belonging to a neighbor octant)

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_global);

    shift_t<dim> shift = get_shift<dim>(0);
    shift[m_direction] = m_cell_block_size[m_direction] * dir;

    const FaceLocation_t face_loc_cur{ face_multiindex_in, key_cur, iOct_global, false };
    const auto           face_loc_neigh = m_stencil_helper.getNeighLoc(face_loc_cur, shift);

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     *
     * There is an extra case when neighbor is outside domain; in that case neighbor is by design
     * at same AMR level and we need border conditions to get involved. We provide here simple
     * border conditions:
     * - wall (copy with sign inversion)
     * - outflow (simple copy)
     *
     * Note that if the mesh is periodic, nothing is needed, by design the neighbor will be inside
     * domain.
     *
     * If a user needs additional border condition for this fluxes, he will have to re-implement
     * this functor.
     */
    const auto iOct_in = face_loc_neigh.iOct;


    if (face_loc_neigh.level() == face_loc_cur.level())
    {

      if (face_loc_neigh.is_outside_domain and
          !m_stencil_helper.is_brick_periodic(face_multiindex_in[dim]))
      {
        // just copy the value from the block border (that is also the domain border)
        // into the ghost
        // TODO: see if we really need more than that

        auto ijk = face_multiindex_in;
        ijk[m_direction] = dir == -1 ? 0 : m_cell_block_size[m_direction];
        const auto flat_index_in = this->to_flat_index(face_multiindex_in, m_fluxes_in.shape());

        m_fluxes_out(flat_index_out, m_var_index_out, iOct_global - m_iOct_begin) =
          m_fluxes_in(flat_index_in, m_var_index_in, iOct_global);
      }
      else
      {
        // doing a simple copy (this is probably not need, cellindex_in computed above is ok)
        const auto flat_index_in = this->to_flat_index(face_loc_neigh.ijk, m_fluxes_in.shape());

        m_fluxes_out(flat_index_out, m_var_index_out, iOct_global - m_iOct_begin) =
          m_fluxes_in(flat_index_in, m_var_index_in, iOct_in);
      }
    }
    else if (face_loc_neigh.level() + 1 == face_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser
      // simple copy of the coarse value

      const auto flat_index_in = this->to_flat_index(face_loc_neigh.ijk, m_fluxes_in.shape());

      m_fluxes_out(flat_index_out, m_var_index_out, iOct_global - m_iOct_begin) =
        m_fluxes_in(flat_index_in, m_var_index_in, iOct_in);
    }
    else if (face_loc_neigh.level() == face_loc_cur.level() + 1)
    {
      // doing a restriction
      m_fluxes_out(flat_index_out, m_var_index_out, iOct_global - m_iOct_begin) =
        m_stencil_helper.compute_face_siblings_average(
          face_loc_neigh, m_cell_block_size, m_var_index_in, m_fluxes_in);
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir == 0)

} // FillBlockGhostFluxesFunctor<dim, device_t>::fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillBlockGhostFluxesFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto nbFluxesPerGhostedLeaf = m_fluxes_out.num_cells();

  const auto iOct_local = global_index / nbFluxesPerGhostedLeaf;
  const auto flat_index_out = global_index - iOct_local * nbFluxesPerGhostedLeaf;
  const auto iOct_global = m_iOct_begin + iOct_local;

  // get face coordinates in the ghost block
  const auto face_multiindex_out =
    to_face_multiindex(flat_index_out, m_fluxes_out.ghosted_block_size(), m_fluxes_out.shift());

  fill_ghosts(flat_index_out, face_multiindex_out, iOct_global);

} // FillBlockGhostFluxesFunctor<dim, device_t>::operator()

template class FillBlockGhostFluxesFunctor<2, kalypsso::DefaultDevice>;
template class FillBlockGhostFluxesFunctor<3, kalypsso::DefaultDevice>;

} // namespace kalypsso
