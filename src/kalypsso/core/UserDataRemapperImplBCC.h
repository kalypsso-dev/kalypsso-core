// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplBCC.h
 *
 * \brief Implement actual user data remapping (after mesh change during AMR cycle) for block
 * cell-centered (BCC) data.
 */
#ifndef KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BCC_H_
#define KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BCC_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost

#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key_utils.h>

#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/prolongation.h>
#include <kalypsso/core/brick_utils.h> // for get_brick_periodicity

namespace kalypsso
{
/**
 * Implementation details class to be instantiated inside UserDataRemapper.
 */
template <size_t dim, typename device_t>
class UserDataRemapperImplBCC
{
public:
  using index_t = int32_t;

  // using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

  using ExecutionSpace = typename device_t::execution_space;

  UserDataRemapperImplBCC() = delete;
  UserDataRemapperImplBCC(const DataArrayBlock_t userdataBlock_old,
                          DataArrayBlock_t       userdataBlock_new,
                          amr_hashmap_t          amr_hashmap_old,
                          orchard_key_view_t     orchard_keys_new,
                          orchard_key_view_t     orchard_keys_old,
                          int32_t                local_num_octants,
                          block_size_t<dim>      bSizes,
                          ConfigMap const &      config_map)
    : m_userdataBlock_old(userdataBlock_old)
    , m_userdataBlock_new(userdataBlock_new)
    , m_amr_hashmap_device_old(amr_hashmap_old)
    , m_orchard_keys_device_new(orchard_keys_new)
    , m_local_num_octants_new(local_num_octants)
    , m_block_sizes(bSizes)
    , m_prolongation(get_cell_prolongation_type(config_map))
    , m_stencil_helper(amr_hashmap_old,
                       orchard_keys_old,
                       userdataBlock_old.block_size(),
                       get_brick_sizes<dim>(config_map),
                       get_brick_periodicity<dim>(config_map))
    , m_cons_interpol()
  {}

  // ==============================================================
  // ==============================================================
  /**
   * Perform remapping in all cells :
   * - level l to level l
   * - fine to coarse (aka restriction)
   * - coarse to fine (aka prolongation)
   */
  KOKKOS_FUNCTION void
  operator()(const index_t & global_index) const;

private:
  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_old,
                                          coord_t<2> const &      coord_new,
                                          index_t const &         cellindex_new,
                                          int32_t const &         iOct_global) const;

  /**
   * Do linear extrapolation using limited slopes.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_old,
                                          coord_t<3> const &      coord_new,
                                          index_t const &         cellindex_new,
                                          int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do second order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order2(CellLocation<2> const & cell_loc_old,
                                    coord_t<2> const &      coord_new,
                                    index_t const &         cellindex_new,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do second order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order2(CellLocation<3> const & cell_loc_old,
                                    coord_t<3> const &      coord_new,
                                    index_t const &         cellindex_new,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do fourth order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order4(CellLocation<2> const & cell_loc_old,
                                    coord_t<2> const &      coord_new,
                                    index_t const &         cellindex_new,
                                    int32_t const &         iOct_global) const;

  // ==============================================================
  // ==============================================================
  /**
   * Do fourth order conservative interpolation from coarse to fine cells.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  conservative_interpolation_order4(CellLocation<3> const & cell_loc_old,
                                    coord_t<3> const &      coord_new,
                                    index_t const &         cellindex_new,
                                    int32_t const &         iOct_global) const;

private:
  //! source data array
  const DataArrayBlock_t m_userdataBlock_old;

  //! destination data array (where data will be remapped)
  DataArrayBlock_t m_userdataBlock_new;

  //! AMR unordered map which map orchard keys to octant number for all key in the mesh
  //! before AMR cycle modification
  amr_hashmap_t m_amr_hashmap_device_old;

  //! list of orchard key of the new mesh (after AMR cycle modification)
  orchard_key_view_t m_orchard_keys_device_new;

  //! number of octants in the new mesh
  int32_t m_local_num_octants_new;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! prolongation type (simple copy or linear extrapolation)
  const CellCenteredProlongationType m_prolongation;

  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! Conservative interpolation data
  const ConservativeInterpolation m_cons_interpol;

}; // class UserDataRemapperImplBCC

// explicit template instantiation
extern template class UserDataRemapperImplBCC<2, kalypsso::DefaultDevice>;
extern template class UserDataRemapperImplBCC<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class UserDataRemapperImplBCC<2, HostDevice>;
extern template class UserDataRemapperImplBCC<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BCC_H_
