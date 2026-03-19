// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AMRCycleDriver.h
 * \brief A driver class to operate a forest through AMR cycle (refine, coarsen, balance).
 *
 * It is also the place where we store additional data to enable userdata remapping before/after AMR
 * cycle (mesh change).
 *
 */
#ifndef KALYPSSO_CORE_AMRCYCLEDRIVER_H_
#define KALYPSSO_CORE_AMRCYCLEDRIVER_H_

#include <kalypsso/core/kalypsso_core_base.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRContext.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

// =============================================================================
// =============================================================================
// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
class AMRCycleDriver
{
public:
  using exec_space = typename device_t::execution_space;

  AMRCycleDriver() = delete;
  AMRCycleDriver(const ParallelEnv & par_env, const ConfigMap & config_map)
    : m_par_env(par_env)
    , m_config_map(config_map)
    , m_amr_mesh(new AMRmesh<dim>(par_env.mpi_comm(), config_map))
    , m_mesh_map(config_map, par_env)
  {

    auto conn_name = m_config_map.getString("amr", "connectivity", "invalid_connectivity");
    if (conn_name == "brick")
    {
      m_brick_sizes[0] =
        static_cast<int>(m_config_map.getInteger("p4est_connectivity", "nbrick_x", 2));
      m_brick_sizes[1] =
        static_cast<int>(m_config_map.getInteger("p4est_connectivity", "nbrick_y", 3));
      if constexpr (dim == 3)
        m_brick_sizes[2] =
          static_cast<int>(m_config_map.getInteger("p4est_connectivity", "nbrick_z", 4));
    }
  } // AMRCycleDriver

  AMRmesh<dim> &
  mesh()
  {
    return *m_amr_mesh;
  }

  const MeshMap<device_t> &
  mesh_map() const
  {
    return m_mesh_map;
  }

  Kokkos::Array<int, 3>
  brick_sizes() const
  {
    return m_brick_sizes;
  }

private:
  //! parallel environment
  const ParallelEnv & m_par_env;

  //! config map (input parameter)
  ConfigMap m_config_map;

  //! p4est resource (forest, ghost, ...)
  std::shared_ptr<AMRmesh<dim>> m_amr_mesh;

  //! a MeshMap object so that we can extract the orchard keys as a kokkos view or unordered map
  MeshMap<device_t> m_mesh_map;

  //! p4est connectivity sizes.
  //! brick sizes are used to transform logical coordinates into vertex space (real space)
  Kokkos::Array<int, 3> m_brick_sizes;

}; // class AMRCycleDriver

} // namespace kalypsso

#endif // KALYPSSO_CORE_AMRCYCLEDRIVER_H_
