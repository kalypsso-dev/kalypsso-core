// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file UserDataRemapperImplBFC.h
 *
 * \brief Implement actual user data remapping (after mesh change during AMR cycle) for block
 * face-centered (BFC) data.
 */
#ifndef KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BFC_H_
#define KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BFC_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for FaceDataArrayBlock

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
class UserDataRemapperImplBFC
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

  //! Compute everything Remapping cell at same AMR level, restriction and only partial prolongation
  //! (only external faces)
  struct TagComputeAllButInternalFaces
  {};

  //! Finalize the remapping by doing prolongation at internal faces
  struct TagComputeInternalFaces
  {};

  UserDataRemapperImplBFC() = delete;
  UserDataRemapperImplBFC(const FaceDataArrayBlock_t facedata_old,
                          FaceDataArrayBlock_t       facedata_new,
                          amr_hashmap_t              amr_hashmap_old,
                          orchard_key_view_t         orchard_keys_new,
                          orchard_key_view_t         orchard_keys_old,
                          int32_t                    local_num_octants,
                          block_size_t<dim>          bSizes,
                          ConfigMap const &          config_map)
    : m_facedata_old(facedata_old)
    , m_facedata_new(facedata_new)
    , m_amr_hashmap_device_old(amr_hashmap_old)
    , m_orchard_keys_device_new(orchard_keys_new)
    , m_local_num_octants_new(local_num_octants)
    , m_block_sizes(bSizes)
    , m_prolongation(ProlongationParam(config_map))
    , m_stencil_helper(amr_hashmap_old,
                       orchard_keys_old,
                       facedata_old.cell_block_size(),
                       get_brick_sizes<dim>(config_map),
                       get_brick_periodicity<dim>(config_map))
  {
    if (m_prolongation.m_face_external ==
        +FaceCenteredProlongationExternalType::EXTRAPOLATE_LINEAR_MINMOD)
    {
      KALYPSSO_ERROR(
        "FaceCenteredProlongationExternalType::EXTRAPOLATE_LINEAR_MINMOD : not yet implemented");
    }
  }

  // ==============================================================
  // ==============================================================
  /**
   * Perform remapping:
   * - restriction
   * - same level
   * - prolongation of external faces
   */
  KOKKOS_FUNCTION void
  operator()(TagComputeAllButInternalFaces const &, const index_t & global_index) const;

  // ==============================================================
  // ==============================================================
  /**
   * Perform remapping:
   * - prolongation of internal faces
   *
   * This is where divergence preserving formulas (e.g. Toth and Roe 2002) are implemented.
   */
  KOKKOS_FUNCTION void
  operator()(TagComputeInternalFaces const &, const index_t & global_index) const;

private:
  /**
   * Do prolongation of external faces by simple copy (injection).
   */
  KOKKOS_INLINE_FUNCTION void
  external_faces_prolongation_by_copy(face_multiindex_t<dim> const & face_indexes_new,
                                      iOct_t const &                 iOct_new,
                                      iOct_t const &                 iOct_old,
                                      uint64_t const &               key_new,
                                      uint64_t const &               key_old,
                                      uint32_t                       family_id) const;

  /**
   * Do prolongation of internal faces using divergence preserving method by Roe and Toth 2002.
   *
   * reference:
   * - "Divergence- and curl-preserving prolongation and restriction formulas.", Toth and Roe, JCP,
   *    180, 746-759, 2002. https://doi.org/10.1006/jcph.2002.7120
   */
  KOKKOS_INLINE_FUNCTION void
  internal_faces_prolongation_by_toth_and_roe(face_multiindex_t<dim> const & face_indexes_new,
                                              iOct_t const &                 iOct_new) const;
  /**
   * Do linear extrapolation using limited slopes on external faces.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<2> const & cell_loc_old,
                                          coord_t<2> const &      coord_new,
                                          index_t const &         cellindex_new,
                                          int32_t const &         iOct_global) const;

  /**
   * Do linear extrapolation using limited slopes on external faces.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  linear_extrapolate_using_limited_slopes(CellLocation<3> const & cell_loc_old,
                                          coord_t<3> const &      coord_new,
                                          index_t const &         cellindex_new,
                                          int32_t const &         iOct_global) const;


private:
  //! source data array
  const FaceDataArrayBlock_t m_facedata_old;

  //! destination data array (where data will be remapped)
  FaceDataArrayBlock_t m_facedata_new;

  //! AMR unordered map which map orchard keys to octant number for all key in the mesh
  //! before AMR cycle modification
  amr_hashmap_t m_amr_hashmap_device_old;

  //! list of orchard key of the new mesh (after AMR cycle modification)
  orchard_key_view_t m_orchard_keys_device_new;

  //! number of octants in the new mesh
  int32_t m_local_num_octants_new;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! prolongation parameter
  const ProlongationParam m_prolongation;

  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

}; // class UserDataRemapperImplBFC

// explicit template instantiation
extern template class UserDataRemapperImplBFC<2, kalypsso::DefaultDevice>;
extern template class UserDataRemapperImplBFC<3, kalypsso::DefaultDevice>;

#ifdef KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE
extern template class UserDataRemapperImplBFC<2, HostDevice>;
extern template class UserDataRemapperImplBFC<3, HostDevice>;
#endif // KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE

} // namespace kalypsso

#endif // KALYPSSO_CORE_USERDATAREMAPPER_IMPL_BFC_H_
