// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file DataWriter.h
 */
#ifndef KALYPSSO_TEST_DATAWRITER_H
#define KALYPSSO_TEST_DATAWRITER_H

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/models/Hydro.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/kalypsso_data_container.h>
#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer_legacy.h>
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif

#include <string>

namespace kalypsso
{

// =============================================================
// =============================================================
template <size_t dim, typename device_t, typename model_t>
struct DataWriter
{
  //! type alias to access p4est C API (2D or 3D)
  using Wrapper = typename p4est::Wrapper<dim>;
  using forest_t = typename Wrapper::forest_t;
  using ghost_t = typename Wrapper::ghost_t;

  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  using DataArrayBlockLegacy_t = DataArrayBlockLegacy<real_t, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

#ifdef KALYPSSO_CORE_USE_HDF5
  using HDF5_Xdmf_Writer_legacy_t = HDF5_Xdmf_Writer_legacy<dim, device_t>;
  using HDF5_Xdmf_Writer_t = HDF5_Xdmf_Writer<dim, device_t>;
#endif

  // ================================================================================
  // ================================================================================
  /**
   * Save both block and leaf data using HDF5_Xdmf_Writer.
   */
  static void
  save(std::string              filename,
       DataArrayLeaf_t          userdataLeaf,
       DataArrayBlock_t         userdataBlock,
       const ConfigMap &        config_map,
       AMRmesh<dim> /*const*/ & amr_mesh,
       model_t const &          model);
  // ================================================================================
  // ================================================================================
  /**
   * Save both block and leaf data using HDF5_Xdmf_Writer.
   */
  static void
  save(std::string                             filename,
       DataArrayLeaf_t                         userdataLeaf,
       DataArrayBlock_t                        userdataBlock,
       const ConfigMap &                       config_map,
       const ParallelEnv &                     par_env,
       std::shared_ptr<MeshMap<dim, device_t>> mesh_map,
       bool                                    use_outside_quads,
       model_t const &                         model);
  // // ================================================================================
  // // ================================================================================
  // /**
  //  * Only save leaf data.
  //  */
  // static void
  // save(std::string              filename,
  //      DataArrayLeaf_t          userdataLeaf,
  //      const ConfigMap &        config_map,
  //      AMRmesh<dim> /*const*/ & amr_mesh,
  //      model_t const &          model);

  // ================================================================================
  // ================================================================================
  /**
   * Save only block data.
   */
  static void
  save(std::string              filename,
       DataArrayBlock_t         userdataBlock,
       const ConfigMap &        config_map,
       /*const*/ AMRmesh<dim> & amr_mesh,
       model_t const &          model);

  // ================================================================================
  // ================================================================================
  /**
   * Save a scalar block data (no field map).
   */
  static void
  save_scalar(std::string              filename,
              DataArrayBlock_t         userdataBlock,
              int                      ivar,
              const ConfigMap &        config_map,
              /*const*/ AMRmesh<dim> & amr_mesh);

  // ================================================================================
  // ================================================================================
  /**
   * Save a ghosted block data.
   */
  static void
  save(std::string              filename,
       DataArrayGhostedBlock_t  userdata_ghosted_block,
       const ConfigMap &        config_map,
       /*const*/ AMRmesh<dim> & amr_mesh,
       bool                     save_full,
       model_t const &          model);

  // ================================================================================
  // ================================================================================
  /**
   * Save only block data of face-centered data.
   */
  static void
  save(std::string              filename,
       FaceDataArrayBlock_t     facedata,
       const ConfigMap &        config_map,
       /*const*/ AMRmesh<dim> & amr_mesh,
       std::string              varname);

}; // struct DataWriter

extern template class DataWriter<2, kalypsso::DefaultDevice, core::models::Hydro>;
extern template class DataWriter<3, kalypsso::DefaultDevice, core::models::Hydro>;

// only instantiate those class when the default device is not on host
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) ||  \
  defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
#else
extern template class DataWriter<2, kalypsso::HostDevice, core::models::Hydro>;
extern template class DataWriter<3, kalypsso::HostDevice, core::models::Hydro>;
#endif
} // namespace kalypsso

#endif // KALYPSSO_TEST_DATAWRITER_H
