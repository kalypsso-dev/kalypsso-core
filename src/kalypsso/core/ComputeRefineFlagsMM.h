// SPDX-FileCopyrightText: 2026 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeRefineFlagsMM.h
 *
 * Implement classic refine indicator compute algorithm, e.g. :
 *
 * - Lohner: [R. Lohner, An adaptive finite element scheme for transient problems in CFD, Comp.
 *           Meth. App. Mech. Eng. 61, 323 (1987)]
 *
 * ComputeRefineFlagsMM is adapted from class ComputeRefineFlags to deal with multimaterial solver
 * data.
 *
 */
#ifndef KALYPSSO_CORE_COMPUTEREFINEFLAGS_MM_H_
#define KALYPSSO_CORE_COMPUTEREFINEFLAGS_MM_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)

#include <kalypsso/core/ComputeRefineFlags_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/StencilHelper.h>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/Kokkos_extensions.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex, and shift_left/shift_right

#include <kalypsso/utils/config/ConfigMap.h>

namespace kalypsso
{

/**
 * \class ComputeRefineFlagsMM
 */
template <size_t dim, typename device_t>
class ComputeRefineFlagsMM
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;
  using CellLocation_t = CellLocation<dim>;

  using amrflag_t = AMRContextBase::amrflag_t;
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  using DataArrayBlockMultiVar_t = DataArrayBlockMultiVar<dim, real_t, device_t>;
  using MaterialPresence_t = MaterialPresenceView<device_t>;

  struct TagLohnerSplit
  {};
  struct TagLohnerUnsplit
  {};
  struct TagSimpleGradient
  {};
  struct TagThresholdAffine
  {};

  // ====================================================================
  // ====================================================================
  //! constructor.
  //!
  //! All variables are inputs, except flags which will be computed
  //!
  //! \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
  //!            quadrants
  //! \param[in] orchard_keys array of orchard key ordered by Morton order
  //! \param[in] local_num_quadrants number of octants (p4est ghosts excluded) owned by current
  //!            MPI process
  //! \param[in] brick_sizes is an array of p4est brick connectivity sizes (number of trees in
  //!            each dimension)
  //! \param[in] is_brick_periodic array of boolean value indicating if the p4est brick
  //!            connectivity is periodic in the given dimension
  //! \param[in] userdata if a block array (owned + MPI ghost blocks) used to compute refinement
  //!            flags
  //! \param[in] mat_pres the material presence array to compute data from
  //! \param[out] flags result of the refinement flags computation
  //! \param[in] refineParams Refinement parameters
  ComputeRefineFlagsMM(amr_hashmap_t            amr_hashmap,
                       orchard_key_view_t       orchard_keys,
                       int32_t                  local_num_octants,
                       brick_size_t<dim>        brick_sizes,
                       Kokkos::Array<bool, dim> is_brick_periodic,
                       DataArrayBlockMultiVar_t userdata,
                       MaterialPresence_t       mat_pres,
                       amrflags_view_t          flags,
                       RefineIndicatorDataMM    refineParams)
    : m_helper(amr_hashmap, orchard_keys, userdata.block_size(), brick_sizes, is_brick_periodic)
    , m_amr_hashmap_device(amr_hashmap)
    , m_orchard_keys_device(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_brick_sizes(brick_sizes)
    , m_is_brick_periodic(is_brick_periodic)
    , m_userdata(userdata)
    , m_mat_pres(mat_pres)
    , m_flags(flags)
    , m_refineParams(refineParams)
  {}

  // ====================================================================
  // ====================================================================
  //! destructor.
  ~ComputeRefineFlagsMM() = default;

  // ====================================================================
  // ====================================================================
  //! run the functor.
  //!
  //! Important note: the caller is responsible for resetting the flags
  //!
  //! \sa ComputeRefineFlagsMM
  static void
  run(amr_hashmap_t            amr_hashmap,
      orchard_key_view_t       orchard_keys,
      int32_t                  local_num_octants,
      brick_size_t<dim>        brick_sizes,
      Kokkos::Array<bool, dim> is_brick_periodic,
      DataArrayBlockMultiVar_t userdata,
      MaterialPresence_t       mat_pres,
      amrflags_view_t          flags,
      RefineIndicatorDataMM    refineParams);

  // ==============================================================
  // ==============================================================
  /**
   * Get, compute or average data in neighbor cell defined as a shift from current cell.
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] shift defines a translation (integer number of cell away from current cell)
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   */
  KOKKOS_FUNCTION real_t
  getNeighborDataSameLevel(CellLocation<dim> const & cell_loc,
                           int32_t                   cell_index,
                           shift_t<dim>              shift,
                           int                       gmat,
                           int                       ivar) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_FUNCTION real_t
  normalized_gradient(real_t const & v1, real_t const & v2) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute Lohner indicator (split version)
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  template <int dir>
  KOKKOS_FUNCTION real_t
  compute_lohner_split(CellLocation_t const & cell_loc,
                       int32_t const &        cell_index,
                       int const &            gmat,
                       int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute Lohner indicator (unsplit version)
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  KOKKOS_FUNCTION real_t
  compute_lohner_unsplit(CellLocation_t const & cell_loc,
                         int32_t const &        cell_index,
                         int const &            gmat,
                         int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute simple gradient refine criterion
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  KOKKOS_FUNCTION real_t
  compute_simple_gradient(CellLocation_t const & cell_loc,
                          int32_t const &        cell_index,
                          int const &            gmat,
                          int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * Compute "threshold affine" refine criterion
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   * \param[in] threshold affine parameters
   *
   * \tparam dir direction along which second partial derivative is computed
   */
  KOKKOS_FUNCTION real_t
  compute_threshold_affine(CellLocation_t const &        cell_loc,
                           int32_t const &               cell_index,
                           int const &                   gmat,
                           int const &                   ivar,
                           ThresholdAffineParams const & th_aff_params) const;

  // ==============================================================
  // ==============================================================
  KOKKOS_FUNCTION void
  update_flag(amrflag_t & flag, real_t const & indicator, uint8_t const & level) const;

  // ==============================================================
  // ==============================================================
  /**
   * Gets the value at a specified location if the material exists else returns 0
   *
   * \param[in] cell_loc is cell location of current cell
   * \param[in] cell_index is cell index of current cell inside current block grid
   * \param[in] gmat global material id
   * \param[in] ivar variable to get
   */
  KOKKOS_FUNCTION real_t
  get_or_zero(CellLocation_t const & cell_loc,
              int32_t const &        cell_index,
              int const &            gmat,
              int const &            ivar) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body - Lohner split version
   */
  KOKKOS_FUNCTION void
  operator()(TagLohnerSplit const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body - Lohner unsplit version
   */
  KOKKOS_FUNCTION void
  operator()(TagLohnerUnsplit const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body - simple gradient version
   */
  KOKKOS_FUNCTION void
  operator()(TagSimpleGradient const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor body - threshold affine version
   */
  KOKKOS_FUNCTION void
  operator()(TagThresholdAffine const &, const index_t & global_index) const;

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

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlockMultiVar_t m_userdata;

  //! material presence
  MaterialPresence_t m_mat_pres;

  //! a leaf data array (one "value" per block)
  amrflags_view_t m_flags;

  //! refine flags algorithm parameters
  RefineIndicatorDataMM m_refineParams;

}; // class ComputeRefineFlagsMM

// explicit template instantiation
extern template class ComputeRefineFlagsMM<2, kalypsso::DefaultDevice>;
extern template class ComputeRefineFlagsMM<3, kalypsso::DefaultDevice>;

} // namespace kalypsso

#endif // KALYPSSO_CORE_COMPUTEREFINEFLAGS_MM_H_
