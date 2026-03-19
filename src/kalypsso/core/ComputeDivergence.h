// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDivergence.h
 *
 * Compute divergence of a vector field represented by cell-centered values.
 *
 * Divergence is evaluated by a simple first order finite difference approximation.
 */
#ifndef KALYPSSO_CORE_COMPUTEDIVERGENCE_H_
#define KALYPSSO_CORE_COMPUTEDIVERGENCE_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/StencilHelper.h>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/Kokkos_extensions.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex

#include <kalypsso/core/models/Hydro.h>

namespace kalypsso
{

/**
 * \class ComputeDivergence
 */
template <size_t dim, typename device_t>
class ComputeDivergence
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using CellLocation_t = CellLocation<dim>;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;

  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  //! All variables are inputs, except flags which will be computed
  //!
  //! \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
  //! quadrants
  //!
  //! \param[in] orchard_keys array of orchard key ordered by Morton order
  //!
  //! \param[in] local_num_quadrants number of octants (p4est ghosts excluded) owned by current
  //! MPI process
  //!
  //! \param[in] block_sizes sizes of the local cartesian grid at leaf level
  //!
  //! \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
  //! each dimension)
  //!
  //! \param[in] is_brick_periodic array of boolean value indicating if the p4est brick
  //! connectivity is periodic in the given dimension
  //!
  //! \param[in] userdata input block array (owned + MPI ghost blocks) used to compute Schlieren
  //! on the first scalar field.
  //!
  //! \param[out] userdata output block array (owned) containing Schlieren of input array
  //!
  ComputeDivergence(amr_hashmap_t            amr_hashmap,
                    orchard_key_view_t       orchard_keys,
                    int32_t                  local_num_octants,
                    block_size_t<dim>        block_sizes,
                    brick_size_t<dim>        brick_sizes,
                    Kokkos::Array<bool, dim> is_brick_periodic,
                    real_t                   scaling_factor,
                    DataArrayBlock_t         userdata_in,
                    DataArrayBlock_t         userdata_out,
                    Kokkos::Array<int, dim>  field_index)
    : m_helper(amr_hashmap, orchard_keys, block_sizes, brick_sizes, is_brick_periodic)
    , m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_nbCellsPerLeaf(Kokkos::dim_prod(block_sizes))
    , m_block_sizes(block_sizes)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_scaling_factor(scaling_factor)
    , m_userdata_in(userdata_in)
    , m_userdata_out(userdata_out)
    , m_field_index(field_index)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~ComputeDivergence() = default;

  // ====================================================================
  // ====================================================================
  //! run the functor.
  //!
  //! \param[in] userdata is a cell-center data array which may contains many field
  //! \param[in] field_index is an array of dim integer to specify which scalar field must be used
  //! to compute divergence
  static auto
  run(amr_hashmap_t            amr_hashmap,
      orchard_key_view_t       orchard_keys,
      int32_t                  local_num_octants,
      block_size_t<dim>        block_sizes,
      brick_size_t<dim>        brick_sizes,
      Kokkos::Array<bool, dim> is_brick_periodic,
      real_t                   scaling_factor,
      DataArrayBlock_t         userdata,
      Kokkos::Array<int, dim>  field_index) -> DataArrayBlock_t;

  // ==============================================================
  // ==============================================================
  /**
   * Get, compute or average data in neighbor cell defined as a shift from current cell.
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] shift defines a translation (integer number of cell away from current cell)
   * \param[in] which variable to get
   */
  KOKKOS_FUNCTION real_t
  getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                           int32_t                   cell_index,
                           shift_t<dim>              shift,
                           int const &               ivar) const;


  // ==============================================================
  // ==============================================================
  /**
   * Compute simple gradient norm
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   */
  KOKKOS_FUNCTION real_t
  compute_divergence(CellLocation_t const & cell_loc, int32_t const & cell_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body
   */
  KOKKOS_FUNCTION void
  operator()(const index_t & global_index) const;

private:
  //! help to compute cell location
  StencilHelper<dim, device_t> m_helper;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! total number of octants in the current MPI process (ghost block excluded)
  const int32_t m_local_num_octants;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! scaling factor
  real_t m_scaling_factor;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata_in;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata_out;

  //! array of index to use to specify x,y and z components
  Kokkos::Array<int, dim> m_field_index;

}; // class ComputeDivergence

extern template class ComputeDivergence<2, kalypsso::DefaultDevice>;
extern template class ComputeDivergence<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEDIVERGENCE_H_
