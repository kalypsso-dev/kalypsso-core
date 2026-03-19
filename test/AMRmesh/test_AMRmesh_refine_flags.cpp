// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh_refine_flags.cpp
 *
 * Purpose: illustrate use of class ComputeRefineFlags.
 *
 * What we do here:
 * - prepare an AMR mesh with some initial refinement (done using regular p4est API, cpu only)
 * - create some user data (as Kokkos device view) attached to the AMR mesh
 * - perform an AMR cycle (refine, coarsen, balance) to modify mesh
 * - remap user data from the old mesh to the new mesh
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/ComputeRefineFlags.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/UserDataRemapper.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshPartitioner.h>
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/core/kalypsso_data_container.h> // for block_size_t
#include <kalypsso/core/models/Hydro.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

namespace kalypsso
{

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{

  using DataArrayLeaf_t = DataArrayLeaf<real_t, device_t>;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  using ExecutionSpace [[maybe_unused]] = typename device_t::execution_space;

  using Hydro_t = core::models::Hydro;

  // auto sigma = config_map.getReal("other", "sigma", KALYPSSO_NUM(0.2));
  //  InitialAMRSetup<dim, device_t, InitGaussian> initial_amr_setup(
  //    par_env, config_map, InitGaussian{ sigma });

  auto radius = config_map.getReal("other", "radius", KALYPSSO_NUM(0.2));
  InitialAMRSetup<dim, device_t, InitHat> initial_amr_setup(par_env, config_map, InitHat{ radius });

  const auto level_min = config_map.getInteger("amr", "level_min", 2);
  const auto level_max = config_map.getInteger("amr", "level_max", 4);
  const auto no_refine = true;

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   forest = amr_mesh.forest();
  auto   mesh_map = initial_amr_setup.mesh_map();
  auto   amr_mesh_info = initial_amr_setup.amr_mesh_info();

  // initialize ghost
  // amr_mesh.reset_ghost();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

  // mirror keys array must be up to date for MeshGhostExchange to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());

  // populate user data
  auto userdata_leaf = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdata_block = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);

  //
  // save data before AMR cycle
  //
  {
    std::string filename =
      dim == 2 ? "test_userdata_2d_before_amr_cycle" : "test_userdata_3d_before_amr_cycle";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_leaf, userdata_block, config_map, amr_mesh, model);
  }

  //
  // here comes the interesting part : apply the AMR cycle
  //


  //
  // 1. store the UnorderedMap before AMR cycle
  //
  // auto       ghost = amr_mesh.ghost();
  const bool on_device = true;
  // mesh_map->template fill_map_serial<dim>(forest, ghost);
  // mesh_map->template fill_map<dim>(forest, ghost, on_device);
  // auto amr_hashmap_device_old = mesh_map->get_map_device();
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device_old = mesh_map->hashmap_clone();

  //
  // 2.1 Update ghost quadrant userdata
  //
#ifdef KALYPSSO_CORE_USE_MPI
  {
    // create the main object doing MPI comm to exchange ghost data
    MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
      config_map, par_env, amr_mesh, *mesh_map);
    MPI_Barrier(par_env.mpi_comm());

    mesh_ghosts_exchanger.exchange(userdata_block);
  }
#endif // KALYPSSO_CORE_USE_MPI

  //
  // 2.2 initalize/compute an amrflags array on device (as we would do in a real application)
  //

  AMRContext<dim, device_t> amr_context(amr_mesh_info.local_num_quadrants());

  const auto refine_threshold = config_map.getReal("amr", "epsilon_refine", KALYPSSO_NUM(0.01));
  const auto coarsen_threshold = config_map.getReal("amr", "epsilon_coarser", KALYPSSO_NUM(0.03));
  const int  ivar_to_refine = 0;

  RefineIndicatorData refine_params{
    level_min,         level_max,      Indicator::LOHNER_UNSPLIT, refine_threshold,
    coarsen_threshold, ivar_to_refine, KALYPSSO_NUM(0.02)
  };
  ComputeRefineFlags<dim, device_t>::run(amr_hashmap_device_old,
                                         orchard_keys_device,
                                         amr_mesh_info.local_num_quadrants(),
                                         initial_amr_setup.brick_sizes(),
                                         initial_amr_setup.is_brick_periodic(),
                                         userdata_block,
                                         amr_context.m_amrflags_d,
                                         refine_params);

  Kokkos::deep_copy(amr_context.m_amrflags_h, amr_context.m_amrflags_d);

  const auto nbCellsPerBlock = userdata_block.num_cells();
  Kokkos::parallel_for(
    "copy_flags",
    Kokkos::RangePolicy<ExecutionSpace>(0,
                                        nbCellsPerBlock * amr_mesh.forest()->local_num_quadrants),
    KOKKOS_LAMBDA(const int32_t i) {
      auto iOct = i / nbCellsPerBlock;
      auto iCell = i - iOct * nbCellsPerBlock;
      userdata_block(iCell, 1, iOct) = static_cast<real_t>(amr_context.m_amrflags_d(iOct));
    });

  //
  // 3. apply AMR cycle on device : refine + coarsen + 2:1 balance
  //
  {
    Kokkos::Profiling::ScopedRegion prof("AMR_refinement_device");
    [[maybe_unused]] auto           changed = amr_context.adapt_mesh(forest);
    KALYPSSO_INFO_ALL("Mesh changed ? {}", changed);
  }

  //
  // re-initialize ghost
  //
  amr_mesh.reset_ghost();
  auto ghost_new = amr_mesh.ghost();

  //
  // prepare for remapping user data
  //
  auto orchard_keys_device_old = mesh_map->orchard_keys_clone();
  mesh_map->update_orchard_keys(amr_mesh.forest(), ghost_new);
  auto orchard_keys_host_new = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device_new = mesh_map->orchard_keys_clone();


  HydroParams params = HydroParams(config_map);
  const int   nbvar = nbvar_hydro<dim>();

  // get block sizes
  const auto bx = config_map.getInteger("amr", "bx", 1);
  const auto by = config_map.getInteger("amr", "by", 1);
  const auto bz = config_map.getInteger("amr", "bz", 1);

  const auto bSizes = [=]() {
    if constexpr (dim == 2)
      return block_size_t<2>{ bx, by };
    if constexpr (dim == 3)
      return block_size_t<3>{ bx, by, bz };
  }();

  // memory allocation for userdata on leaf after remapping
  auto userdata_leaf_new = DataArrayLeaf_t("test_data_after_amr_cycle",
                                           static_cast<size_t>(forest->local_num_quadrants),
                                           static_cast<size_t>(nbvar));

  // memory allocation for userdata on block after remapping
  auto userdata_block_new =
    DataArrayBlock_t("test_data_block_after_amr_cycle", bSizes, nbvar, forest->local_num_quadrants);

  // retrieve the new local number of octants (after AMR cycle)
  auto local_num_octants = forest->local_num_quadrants;

  // const bool on_device = true;
  //  rebuild the new hashmap (after AMR cycle)
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device_new = mesh_map->hashmap_clone();

  KALYPSSO_INFO_ALL("after create map on device");

  ExecutionSpace exec_space;

  UserDataRemapper<dim, device_t> userdataRemapper(amr_hashmap_device_old,
                                                   orchard_keys_device_new,
                                                   orchard_keys_device_old,
                                                   local_num_octants,
                                                   bSizes,
                                                   config_map);

  //
  // 4.1 apply data remapping (on device) using the hash map approach - leaf data
  //
  {
    KALYPSSO_INFO_ALL("after AMR local_num_quadrants = {} - remapping leaf data",
                      forest->local_num_quadrants);

    userdataRemapper.remap_leaf_data(exec_space, userdata_leaf, userdata_leaf_new);
  } // end apply remapping

  //
  // 4.2 apply data remapping (on device) using the hash map approach - cell data
  //
  {
    KALYPSSO_INFO_ALL("after AMR local_num_quadrants = {} - remapping cell data",
                      forest->local_num_quadrants);
    userdataRemapper.remap_block_data(exec_space, userdata_block, userdata_block_new);
  } // end apply remapping

  KALYPSSO_INFO_ALL("after remapping");

  //
  // save data after AMR cycle
  //
  {
    std::string filename =
      dim == 2 ? "test_userdata_2d_after_amr_cycle" : "test_userdata_3d_after_amr_cycle";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_leaf_new, userdata_block_new, config_map, amr_mesh, model);
  }

#ifdef KALYPSSO_CORE_USE_MPI
  //
  // perform load balance, aka mesh re-partitioning
  //
  {
    MeshPartitioner<dim, device_t> mesh_partitioner(config_map, par_env);

    // after that, forest will change
    mesh_partitioner.partition_mesh(forest);

    // repartition user data

    auto userdata_block_new2 = DataArrayBlock_t(
      "test_data_block_after_partition", bSizes, nbvar, forest->local_num_quadrants);

    mesh_partitioner.repartition_userdata(userdata_block_new, userdata_block_new2);

    auto userdata_leaf_new2 = DataArrayLeaf_t("test_data_leaf_after_partition",
                                              static_cast<size_t>(forest->local_num_quadrants),
                                              static_cast<size_t>(nbvar));

    mesh_partitioner.repartition_userdata(userdata_leaf_new, userdata_leaf_new2);


    //
    // save data after re-partitioning
    //
    {
      std::string filename =
        dim == 2 ? "test_userdata_2d_after_partition" : "test_userdata_3d_after_partition";

      // provide mapping between variables Id and variables names
      const Hydro_t model(dim);

      // when using actually only one MPI rank, MeshPartioner didn't do anything,
      // so we need to chose which array to save
      DataWriter<dim, device_t, Hydro_t>::save(
        filename,
        par_env.nRanks() > 1 ? userdata_leaf_new2 : userdata_leaf_new,
        par_env.nRanks() > 1 ? userdata_block_new2 : userdata_block_new,
        config_map,
        amr_mesh,
        model);
    }

  } // end load balancing
#endif // KALYPSSO_CORE_USE_MPI

} // run_test

} // namespace kalypsso

bool
hasEnding(std::string const & fullString, std::string const & ending)
{
  if (fullString.length() >= ending.length())
  {
    return (0 ==
            fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else
  {
    return false;
  }
} // hasEnding

// =============================================================================
// =============================================================================
// =============================================================================
int
main(int argc, char ** argv)
{

  {
    // initialize mpi, kokkos and p4est, use default MPI communicator : MPI_COMM_WORLD
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 1 ./test_AMRmesh_refine_flags --ini test_AMRmesh_brick_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // parse command line : 2d or 3d ?
    bool threed = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

    // check if user passed a custom ini filename
    std::string config_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    // provide a default config filename (that exists)
    if (config_filename.size() == 0)
      config_filename = threed ? "./test_AMRmesh_brick_3d.ini" : "./test_AMRmesh_brick_2d.ini";

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_AMRmesh_refine_flags] Wrong dimension");

    // check command line for optional parameter (refine_level)
    kalypsso::InitialAMRSetupBase::refine_level =
      kalypsso::cmdline_arg_exists(argv, argv + argc, "--refine_level")
        ? kalypsso::cmdline_get_integer(argv, argv + argc, "--refine_level")
        : 4;

    if (par_env.rank() == 0)
    {
      printf("================================================\n");
      printf("START RUN_TEST\n");
      printf("================================================\n");
    }

    if (dim == 2)
    {
      kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else if (dim == 3)
    {
      kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else
    {
      if (par_env.rank() == 0)
      {
        std::cout << "Input file is not valid ! check parameter run/dimension.\n";
      }
    }

    if (par_env.rank() == 0)
    {
      printf("================================================\n");
      printf("END RUN_TEST\n");
      printf("================================================\n");
    }
  }

  return 0;

} // main
