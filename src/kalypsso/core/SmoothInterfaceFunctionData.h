// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file SmoothInterfaceFunctionData.h
 * \brief Container for data needed in smooth interface function related algorithms.
 */
#ifndef KALYPSSO_CORE_SMOOTH_INTERFACE_FUNCTION_DATA_H_
#define KALYPSSO_CORE_SMOOTH_INTERFACE_FUNCTION_DATA_H_

#include <kalypsso/core/kalypsso_core_base.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock, FaceDataArrayBlock, ...
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>

#include <kalypsso/core/config_utils.h> // for get_block_sizes
#include <kalypsso/core/InterfaceNormalVectorAlgorithmParams.h>

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

// AMR services
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

namespace kalypsso
{

// ========================================================================================
// ========================================================================================
// ========================================================================================
/**
 * \class SmoothInterfaceFunctionData
 *
 * \brief A container class for all data used in smooth interface function related algorithms.
 *
 * Algorithms are:
 * - compute the smooth interface function psi
 * - compute interface normal vector (gradient of psi) using a fourth order central difference
 *   scheme
 * - compute surface tension
 *
 * Note that interface normal vector and curvature are stored in a DataArrayGhostedBlock but only
 * the inner part is computed here. We chose to use DataArrayGhostedBlock instead of DataArrayBlock
 * to ease the use of these data in downstream code (e.g. multifluid Riemann solver).
 * We use a ghostwidth of 1 all around (TODO: see if ghostwidth needs to be a configuration
 * variable).
 *
 * \tparam dim is dimension (integer: 2 or 3)
 * \tparam device_t is a kokkos device class (e.g. Kokkos::CudaSpace::device_type)
 */
template <size_t dim, typename device_t>
class SmoothInterfaceFunctionData
{
public:
  using exec_space = typename device_t::execution_space;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  static constexpr int NUM_VALUES = 1;

  // =========================================================================
  // =========================================================================
  //! constructor
  SmoothInterfaceFunctionData([[maybe_unused]] ParallelEnv const &      par_env,
                              ConfigMap const &                         config_map,
                              [[maybe_unused]] AMRmesh<dim> &           amr_mesh,
                              [[maybe_unused]] MeshMap<dim, device_t> & mesh_map)
    : m_config_map(config_map)
    , m_enabled(config_map.getBool("smooth_interface_function", "enabled", false))
    , m_surface_tension_enabled(
        config_map.getBool("smooth_interface_function", "surface_tension_enabled", false))
    , m_block_sizes(get_block_sizes<dim>(config_map))
    , m_sif()
    , m_normal_vector()
    , m_curvature_weights()
    , m_unfiltered_curvature()
    , m_curvature()
    , m_mesh_map(mesh_map)
#ifdef KALYPSSO_CORE_USE_MPI
    , m_mesh_ghosts_exchanger(config_map, par_env, amr_mesh, mesh_map)
#endif // KALYPSSO_CORE_USE_MPI
  {

    if (m_enabled)
    {
      m_sif = DataArrayGhostedBlock_t(m_block_sizes,
                                      m_block_sizes + 2 * 0,
                                      get_shift<dim>(0),
                                      "smooth_interface_function",
                                      NUM_VALUES,
                                      0);
      m_normal_vector = DataArrayGhostedBlock_t(
        m_block_sizes, m_block_sizes + 2 * 1, get_shift<dim>(-1), "normal_vector", dim, 0);

      if (m_surface_tension_enabled)
      {
        m_curvature_weights = DataArrayGhostedBlock_t(m_block_sizes,
                                                      m_block_sizes + 2 * 1,
                                                      get_shift<dim>(-1),
                                                      "curvature_weights",
                                                      NUM_VALUES,
                                                      0);

        m_unfiltered_curvature = DataArrayGhostedBlock_t(m_block_sizes,
                                                         m_block_sizes + 2 * 1,
                                                         get_shift<dim>(-1),
                                                         "unfiltered curvature",
                                                         NUM_VALUES,
                                                         0);

        m_curvature = DataArrayGhostedBlock_t(
          m_block_sizes, m_block_sizes + 2 * 1, get_shift<dim>(-1), "curvature", NUM_VALUES, 0);
      }
    }

  } // SmoothInterfaceFunctionData::SmoothInterfaceFunctionData

  // =========================================================================
  // =========================================================================
  //! constructor
  ~SmoothInterfaceFunctionData() = default;

  // =========================================================================
  // =========================================================================
  auto
  enabled() const
  {
    return m_enabled;
  }

  // =========================================================================
  // =========================================================================
  auto
  surface_tension_enabled() const
  {
    return m_surface_tension_enabled;
  }

  // =========================================================================
  // =========================================================================
  //! Resize our workspace data.
  //!
  //! \param[in] num_octants new number of octants
  void
  resize();

  // =========================================================================
  // =========================================================================
  //! memory footprint monitoring
  uint64_t
  total_mem_size_in_bytes() const;

  // =========================================================================
  // =========================================================================
  //! compute smooth interface function (SIF)
  void
  compute_smooth_interface_function(DataArrayBlock_t const & userdata_in, int32_t ivar);

  // =========================================================================
  // =========================================================================
  //! compute interface normal vector
  void
  compute_interface_normal_vector(const OrchardKeys & keys);

  // =========================================================================
  // =========================================================================
  //! compute curvature
  //!
  //! \param[in] keys Orchard keys.
  //! \param[in] qdata Array of primitive variables
  //! \param[in] iphi index to volume fraction
  void
  compute_curvature(const OrchardKeys & keys, DataArrayGhostedBlock_t const & qdata, int32_t iphi);

  //
  // Data members
  //

  //! config map
  const ConfigMap & m_config_map;

  //! smooth interface function enabled ?
  bool m_enabled;

  //! surface tension enabled ?
  bool m_surface_tension_enabled;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! smooth interface function
  DataArrayGhostedBlock_t m_sif;

  //! interface normal vector (required for doing THINC reconstruction)
  DataArrayGhostedBlock_t m_normal_vector;

  //! curvature weights (optional) used in filtering step
  DataArrayGhostedBlock_t m_curvature_weights;

  //! unfiltered curvature (optional)
  DataArrayGhostedBlock_t m_unfiltered_curvature;

  //! curvature (optional)
  DataArrayGhostedBlock_t m_curvature;

  //! mesh map is a helper class for accessing orchard keys
  MeshMap<dim, device_t> & m_mesh_map;

#ifdef KALYPSSO_CORE_USE_MPI
  //! MPI communications to exchange ghost block userdata
  MeshGhostsExchanger<dim, real_t, device_t> m_mesh_ghosts_exchanger;
#endif // KALYPSSO_CORE_USE_MPI

}; // class SmoothInterfaceFunctionData

} // namespace kalypsso

#endif // KALYPSSO_CORE_SMOOTH_INTERFACE_FUNCTION_DATA_H_
