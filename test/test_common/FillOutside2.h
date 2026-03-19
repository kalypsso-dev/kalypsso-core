// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside2.h
 *
 * Implement border conditions (other than periodic) for Hydrodynamics.
 *
 * Same as FillOutside but using DataArrayGhostedBlock and filling only the inner part of the block
 * of cells.
 */
#ifndef KALYPSSO_TEST_FILLOUTSIDE2_H_
#define KALYPSSO_TEST_FILLOUTSIDE2_H_

#include <kalypsso/core/FillOutside_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/init_func.h> // for InitFunc1 and InitFunc2

#include <kalypsso/core/models/Hydro.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace test
{

/**
 * Define a (better) enum to list supported boundary conditions.
 *
 * - NONE: don't fill this border
 * - PERIODIC : don't fill this border (it will be filled by MPI ghost exchange operations).
 * - ZERO_GRADIENT: it means here that we fill outside cells with a copy of the "last" cell inside
 *                  domain.
 * - WALL : it means here that we fill outside cells with wall boundary condition (normal
 *          velocity is negated, other variable are copied using planar symmetry; at edge we use
 *          axial symmetry and at corners, central symmetry).
 * - BC_CUSTOM : provide a pointwise functor (TODO)
 *
 * \todo should we consider :
 *  - DIRICHLET (constant value at border) ?
 *  - NEUMANN (constant gradient at border) ?
 *
 * \note a (better) enum cannot be defined inside a class (that is a bit annoying)
 */
// clang-format off
BETTER_ENUM(BC_HYDRO, uint32_t,
  NONE,
  PERIODIC,
  ZERO_GRADIENT,
  WALL,
  ANALYTICAL,
  CUSTOM
  )
// clang-format on

// ==============================================================================
// ==============================================================================
/**
 * \class FillOutside2CellFunctor
 *
 * A simple prototyping boundary condition class.
 *
 * \note periodic boundary condition is a special case that is not managed here.
 * \note in a given direction, either all opposite sides are periodic, or none.
 *
 * \tparam AnalyticalBC must be a functor with 2 operators(), so that, if "f" is an instance of
 *         AnalyticalBC, then f(x,y,var) and f(x,y,z,var) will provide analytical values (for both
 *         2d and 3D)
 *
 * How are filled cells in corner (2d) ?
 * corner are currently filled twice, border condition along Y border prevail X border.
 * Same for 3D.
 *
 * --------------------------------
 * |              3               |
 * |______________________________|
 * |    |                    |    |
 * |    |                    |    |
 * | 1  |                    | 2  |
 * |    |                    |    |
 * |    |                    |    |
 * |____|____________________|____|
 * |              4               |
 * |______________________________|
 *
 * In you need a border condition with more control of how corner blocks are filled, you need to
 * modify/customize this prototype border condition functor.
 */
template <size_t dim, typename device_t, typename AnalyticalBC = AnalyticalZeroBC>
class FillOutside2CellFunctor
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  using bc_array_t = BorderConditionsConfig<BC_HYDRO>::bc_array_t<dim>;

  /**
   *
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees
   *            in each dimension)
   * \param[in] boundary condition type (for each boundary: xmin, xmax, ymin, ymax, ...)
   *
   * This constructor only specify BC type for faces. Corners (and edges) will be filled with zero
   * gradient (default).
   *
   * \todo: add another constructor containing BC type for all geometrical elements (faces, edges
   * and corners). This should be think through.
   *
   */
  FillOutside2CellFunctor(ConfigMap const &        config_map,
                          amr_hashmap_t            amr_hashmap,
                          orchard_key_view_t       orchard_keys,
                          AMRMeshInfo              amr_mesh_info,
                          DataArrayGhostedBlock_t  userdata,
                          block_size_t<dim>        block_sizes,
                          brick_size_t<dim>        brick_sizes,
                          Kokkos::Array<bool, dim> is_brick_periodic,
                          bc_array_t               bc_types,
                          AnalyticalBC const &     f = AnalyticalBC{});

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        AMRMeshInfo              amr_mesh_info,
        DataArrayGhostedBlock_t  userdata,
        block_size_t<dim>        block_sizes,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic,
        bc_array_t               bc_types);

  // ==============================================================
  // ==============================================================
  //! special case: apply analytical border condition specified with a functor
  //!
  static void
  apply(ConfigMap const &        config_map,
        amr_hashmap_t            amr_hashmap,
        orchard_key_view_t       orchard_keys,
        AMRMeshInfo              amr_mesh_info,
        DataArrayGhostedBlock_t  userdata,
        block_size_t<dim>        block_sizes,
        brick_size_t<dim>        brick_sizes,
        Kokkos::Array<bool, dim> is_brick_periodic,
        bc_array_t               bc_types,
        const AnalyticalBC &     f);

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor to sweep outside leaves.
   *
   * Just explaining inside/outside mirroring.
   *                       _
   *  ____________________|1|________
   * |                    |2|        |
   * |                               |
   * |                               |
   * |                     _         |
   * |____________________|3|________|
   *                      |4|
   *
   * Reminder:
   *
   * - 1 and 4 are outside quadrants
   * - 2 and 3 are inside  quadrants
   * - 1 and 3 have the same orchard key, except the outside status bits
   * - 2 and 4 have the same orchard key, except the outside status bits
   *
   * Suppose we have orchard key of 1 (say key1) and we want to compute orchard key of quadrant 2:
   *
   * 1. from key1, compute key3 (just change outside status bits)
   * 2. compute outside normal from key3 (which the same as outside normal of key1, since key1==key3
   *    and compute normal does not depend on outside status)
   * 3. use outside normal to compute key4 from key3
   * 4. deduce key2 from key4 (same orchard key, different outside bit status)
   *
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index) const;

private:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh information (number of owned quadrants, number of ghosts quadrants, number of
  //! outside quads, number of outside ghosts, etc...)
  AMRMeshInfo m_amr_mesh_info;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayGhostedBlock_t m_userdata;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! foer each direction, state if mesh is periodic
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! border conditions array (one for each face, edge, corner)
  const bc_array_t m_bc_types;

  //! only used when doing analytical border conditions (only really useful for test and debug)
  const AnalyticalBC & m_f;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

}; // class FillOutside2CellFunctor

// explicit template instantiation
extern template class FillOutside2CellFunctor<2, kalypsso::DefaultDevice>;
extern template class FillOutside2CellFunctor<3, kalypsso::DefaultDevice>;

// this may not be handy:
// we have to declare here, and instantiate in FillOutside2.cpp all class for a particular desired
// init functor.
extern template class FillOutside2CellFunctor<2, kalypsso::DefaultDevice, InitFunc1>;
extern template class FillOutside2CellFunctor<3, kalypsso::DefaultDevice, InitFunc1>;
extern template class FillOutside2CellFunctor<2, kalypsso::DefaultDevice, InitFunc2>;
extern template class FillOutside2CellFunctor<3, kalypsso::DefaultDevice, InitFunc2>;

} // namespace test

} // namespace kalypsso

#endif // KALYPSSO_TEST_FILLOUTSIDE2_H_
