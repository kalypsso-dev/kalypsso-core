// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MirrorQuadrantUtilsMetadata_test.cpp
 *
 * unit test for class MirrorQuadrantUtilsMetadata.
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/real_type.h>

#include <kalypsso/core/MirrorQuadrantUtilsMetadata.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

#include "../main_kalypsso_unittest.h"

#include "gtest/gtest.h"

namespace kalypsso
{

// =============================================================================
// =============================================================================
template <int dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{
  using Hydro_t = core::models::Hydro;

  InitialAMRSetup<dim, device_t, InitFuncSineWave> initial_amr_setup(
    par_env, config_map, InitFuncSineWave{});

  const auto level_min = config_map.getInteger("amr", "level_min", 2);
  const auto level_max = config_map.getInteger("amr", "level_max", 2);
  const auto no_refine = (level_min == level_max);

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   mesh_map = initial_amr_setup.mesh_map();
  auto   block_sizes = initial_amr_setup.block_sizes();
  auto   brick_sizes = initial_amr_setup.brick_sizes();
  auto   is_brick_periodic = initial_amr_setup.is_brick_periodic();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();
  auto orchard_keys_mirror = mesh_map->mirror_orchard_keys();

  // mirror keys array must be up to date for MeshGhostExchange to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());

  // rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device = mesh_map->hashmap();
  mesh_map->update_conformal_status();
  auto conformal_status = mesh_map->conformal_status();


  auto amr_mesh_info = initial_amr_setup.amr_mesh_info();

  auto mirror_quadrant_utils_metadata = core::MirrorQuadrantUtilsMetadata<dim, device_t>(par_env);

  mirror_quadrant_utils_metadata.update(*mesh_map, amr_mesh, config_map);

  {
    // for testing DataArrayBlock
    auto userdata_block_cell = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);

    std::string filename = dim == 2 ? "StencilHelper_test_2d" : "StencilHelper_test_3d";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_block_cell, config_map, amr_mesh, model);
  }

  if constexpr (dim == 2)
  {
    if (par_env.rank() == 0)
    {
      const uint32_t iOct = 1284;
      for (int mirror_id = 0; mirror_id < orchard_keys_mirror.size(); ++mirror_id)
      {
        auto global_index = mirror_quadrant_utils_metadata.mirror_neighbor_index(mirror_id, 0, 0);
        // if (global_index == iOct)
        {
          for (int j = -1; j < 2; ++j)
            for (int i = -1; i < 2; ++i)
              printf("mirror_id = %d global_id=%ld mirror_rank=%d dir=%d,%d neighbor_index=%ld "
                     "active=%d\n",
                     mirror_id,
                     mirror_quadrant_utils_metadata.mirror_neighbor_index(mirror_id, 0, 0),
                     mirror_quadrant_utils_metadata.mirror_quadrant_mpi_ranks()(mirror_id),
                     i,
                     j,
                     mirror_quadrant_utils_metadata.mirror_neighbor_index(mirror_id, i, j),
                     mirror_quadrant_utils_metadata.active_neigbor(mirror_id, i, j));
        }
      }
    }
  } // dim ==2
  else if constexpr (dim == 3)
  {
    // if (par_env.rank() == 0)
    // {
    //   for (int mirror_id = 0; mirror_id < orchard_keys_mirror.size(); ++mirror_id)
    //   {
    //     printf("mirror_id = %d global_id=%ld\n",
    //            mirror_id,
    //            mirror_quadrant_utils_metadata.mirror_neighbor_index(mirror_id, 0, 0));
    //   }
    // }
  } // dim == 3

} // run_test

// ====================================================================
// ====================================================================
TEST(kalypsso_shared, mirror_quadrant_utils_metadata)
{

  // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
  ParallelEnv & par_env = *(*g_par_env_ptr);

#ifdef KALYPSSO_CORE_USE_SPDLOG
  kalypsso_spdlog_config(*global_argc, *global_argv, par_env.rank(), par_env.size());
#endif

  {
    // run a 2d test
    ConfigMap config_map = broadcast_parameters("./StencilHelper_2d.ini");
    kalypsso::run_test<2, kalypsso::HostDevice>(par_env, config_map);
  }

  // TODO make 3d mesh smaller (currently much too large for unit testing)
  {
    // run a 3d test
    ConfigMap config_map = broadcast_parameters("./StencilHelper_3d.ini");
    kalypsso::run_test<3, kalypsso::HostDevice>(par_env, config_map);
  }

} // TEST

} // namespace kalypsso
