// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeSchlieren.h
 *
 * Compute schlieren of a scalar field.
 *
 */
#ifndef KALYPSSO_CORE_COMPUTESCHLIEREN_H_
#define KALYPSSO_CORE_COMPUTESCHLIEREN_H_

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

#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

namespace core
{

// clang-format off
BETTER_ENUM(SCHLIEREN_SCALING, uint32_t,
  NONE,
  SQRT,
  LOG
)
// clang-format on


/**
 * \class ComputeSchlieren
 */
template <size_t dim, typename device_t>
class ComputeSchlieren
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
  ComputeSchlieren(amr_hashmap_t            amr_hashmap,
                   orchard_key_view_t       orchard_keys,
                   int32_t                  local_num_octants,
                   block_size_t<dim>        block_sizes,
                   brick_size_t<dim>        brick_sizes,
                   Kokkos::Array<bool, dim> is_brick_periodic,
                   real_t                   scaling_factor,
                   DataArrayBlock_t         userdata_in,
                   DataArrayBlock_t         userdata_out)
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
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~ComputeSchlieren() = default;

  // ====================================================================
  // ====================================================================
  //! run the functor.
  //!
  static auto
  run(ConfigMap const &        config_map,
      const ParallelEnv &      par_env,
      amr_hashmap_t            amr_hashmap,
      orchard_key_view_t       orchard_keys,
      int32_t                  local_num_octants,
      block_size_t<dim>        block_sizes,
      brick_size_t<dim>        brick_sizes,
      Kokkos::Array<bool, dim> is_brick_periodic,
      DataArrayBlock_t         userdata) -> DataArrayBlock_t;

  // ====================================================================
  // ====================================================================
  static SCHLIEREN_SCALING
  read_schlieren_scaling(ConfigMap const & config_map)
  {

    const auto schlieren_type_str = config_map.getString("output", "schlieren_scaling", "NONE");

    // check if schlieren_type_str is a valid value
    auto maybe_value = SCHLIEREN_SCALING::_from_string_nothrow(schlieren_type_str.c_str());
    if (!maybe_value)
    {
      return SCHLIEREN_SCALING::NONE;
    }
    else
    {
      return *maybe_value;
    }
  } // read_schlieren_scaling

  // ====================================================================
  // ====================================================================
  /**
   * Get, compute or average data in neighbor cell defined as a shift from current cell.
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] shift defines a translation (integer number of cell away from current cell)
   * \param[in] which variable to get
   * \param[out] is_neighbor_owned is status to inform caller
   */
  KOKKOS_FUNCTION real_t
  getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                           int32_t                   cell_index,
                           shift_t<dim>              shift,
                           int                       ivar,
                           bool &                    is_neighbor_not_owned) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute first derivative along direction dir using a 3 point stencil.
   *
   * Derivative is estimated with big O(h^2) approximation.
   */
  template <int32_t dir>
  KOKKOS_FUNCTION real_t
  compute_first_derivative_3_points(CellLocation<dim> const & cell_loc, int ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute gradient along dir
   */
  template <int32_t dir>
  KOKKOS_FUNCTION real_t
  compute_gradient_along_dir(CellLocation_t const & cell_loc,
                             int32_t const &        cell_index,
                             int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute simple gradient norm
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] ivar which variable to get
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  KOKKOS_FUNCTION real_t
  compute_norm_of_gradient(CellLocation_t const & cell_loc,
                           int32_t const &        cell_index,
                           int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body
   */
  KOKKOS_FUNCTION void
  operator()(const index_t & global_index, real_t & max_norm_grad) const;

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

}; // class ComputeSchlieren

// explicit template instantiation
extern template class ComputeSchlieren<2, kalypsso::DefaultDevice>;
extern template class ComputeSchlieren<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTESCHLIEREN_H_
