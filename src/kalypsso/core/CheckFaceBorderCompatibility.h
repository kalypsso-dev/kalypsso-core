// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file CheckFaceBorderCompatibility.h
 *
 * The purpose of this functor is to check that a FaceDataArrayBlock is continuous at block border.
 *
 * There are two possible situations:
 *
 * - when two neighbor blocks live at the same AMR level, we just check equality of the face data
 * array values on the interface.
 * - when two neighbor blocks are at different AMR levels, the interface is non-conform (see drawing
 * below), then we check that the values on the large block face (Block A) is equal to the average
 * of the values on the small faces of block B.
 *
 *
 *  Block A
 *  ____________________________
 * |      |      |      |      |
 * |      |      |      |      |
 * |      |      |      |      |
 * |______|______|______|______|
 * |      |      |      |      |
 * |      |      |      |      |
 * |      |      |      |      |              Block B
 * |______|______|______|______|              ________________
 * |      |      |      |      |         ____|   |   |   |   |
 * |      |      |      |      |______ /     |___|___|___|___|
 * |      |      |      |      |       \_____|   |   |   |   |
 * |______|______|______|______|             |___|___|___|___|
 * |      |      |      |      |             |   |   |   |   |
 * |      |      |      |      |             |___|___|___|___|
 * |      |      |      |      |             |   |   |   |   |
 * |______|______|______|______|             |___|___|___|___|
 *
 *
 */
#ifndef KALYPSSO_CORE_CHECKFACEBORDERCOMPATIBILITY_H_
#define KALYPSSO_CORE_CHECKFACEBORDERCOMPATIBILITY_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>

#include <kalypsso/core/DataArrayBlock_utils.h>
#include <kalypsso/core/ConformalFaceStatus.h>

#include <type_traits>

namespace kalypsso
{

namespace core
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Check that a FaceDataArrayBlock is continuous at block border.
 *
 */
template <size_t dim, typename device_t>
class CheckFaceBorderCompatibility
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using CellLocation_t = CellLocation<dim>;
  using FaceLocation_t = FaceLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! some face array (in)
  FaceDataArrayBlock_t m_facedata;

  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  const AMRMeshInfo m_amr_mesh_info;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! threshold
  KALYPSSO_STATIC_MATH_CONSTANT(SMALL, 1e-13);

  //! MPI comm rank from parallel environment
  const int m_mpi_comm_rank;

public:
  /**
   * \param[in]  facedata A FaceDataArrayBlock to check.
   * \param[in]  config_map A config map.
   * \param[in]  stencil_helper A stencil helper object.
   * \param[in]  orchard_keys Orchard keys.
   * \param[in]  amr_mesh_info AMR mesh info.
   *
   */
  CheckFaceBorderCompatibility(FaceDataArrayBlock_t const & facedata,
                               StencilHelper_t const &      stencil_helper,
                               orchard_key_view_t const &   orchard_keys,
                               AMRMeshInfo const &          amr_mesh_info,
                               const int                    mpi_comm_rank,
                               ConfigMap const &            config_map);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(FaceDataArrayBlock_t const &     facedata,
        amr_hashmap_t const &            amr_hashmap,
        orchard_key_view_t const &       orchard_keys,
        AMRMeshInfo const &              amr_mesh_info,
        ConfigMap const &                config_map,
        brick_size_t<dim> const &        brick_sizes,
        Kokkos::Array<bool, dim> const & is_brick_periodic,
        ParallelEnv const &              par_env);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  check(index_t const & surface_flatindex, int32_t const & iOct) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // CheckFaceBorderCompatibility

// explicit template instantiation
extern template class CheckFaceBorderCompatibility<2, kalypsso::DefaultDevice>;
extern template class CheckFaceBorderCompatibility<3, kalypsso::DefaultDevice>;

} // namespace core

} // namespace kalypsso

#endif // KALYPSSO_CORE_CHECKFACEBORDERCOMPATIBILITY_H_
