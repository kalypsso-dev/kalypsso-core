// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRContext.cpp
 *
 * Purpose: illustrate use of class AMRContext (helper class provide all that is necessary to
 * perform AMR cycle on device, GPU compatible).
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
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/UserDataRemapper.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshPartitioner.h>
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/core/kalypsso_data_container.h> // for block_size_t

#include <test_common/InitialAMRSetup.h>
#include <kalypsso/core/models/Hydro.h>
#include <test_common/DataWriter.h>

#ifdef KALYPSSO_CORE_USE_CNPY
#  include <kalypsso/core/cnpy_io.h>
#endif // KALYPSSO_CORE_USE_CNPY

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

  using FaceDataArrayBlock_t = FaceDataArrayBlock<dim, real_t, device_t>;

  using ExecutionSpace = typename device_t::execution_space;

  using Hydro_t = core::models::Hydro;

  InitialAMRSetup<dim, device_t, InitFunc3> initial_amr_setup(par_env, config_map, InitFunc3{});

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh();

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   forest = amr_mesh.forest();
  auto   mesh_map = initial_amr_setup.mesh_map();

  // initialize ghost
  amr_mesh.reset_ghost();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host();
  auto orchard_keys_device = mesh_map->orchard_keys();

#ifdef KALYPSSO_CORE_USE_CNPY
  // save orchard key corresponding to locally owned quadrants (exclude ghost quadrant)
  save_cnpy(
    Kokkos::subview(orchard_keys_host,
                    std::make_pair(0, initial_amr_setup.mesh().forest()->local_num_quadrants)),
    "orchard_keys_before_amr",
    par_env);
#endif


  // populate user data
  auto userdataLeaf = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdataBlock = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);
  auto userdataBlock_face =
    initial_amr_setup.setup_initial_data_block_face(orchard_keys_device, false);

  ExecutionSpace exec_space;

  //
  // save data before AMR cycle
  //
  {
    std::string filename =
      dim == 2 ? "test_userdata_2d_before_amr_cycle" : "test_userdata_3d_before_amr_cycle";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdataLeaf, userdataBlock, config_map, amr_mesh, model);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename + "_face", userdataBlock_face, config_map, amr_mesh, "facedata");

    // compute divergence of the face array
    const auto divBdata =
      FaceDataArrayBlock_t::compute_divergence(userdataBlock_face, mesh_map->orchard_keys());

    DataWriter<dim, device_t, Hydro_t>::save_scalar(
      filename + std::string("_face_div"), divBdata, 0, config_map, amr_mesh);
  }

  //
  // here comes the interesting part : apply the AMR cycle
  //


  //
  // 1. store the UnorderedMap before AMR cycle
  //
  const bool on_device = true;
  // update hashmap (orchard keys array are already up to date)
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device_old = mesh_map->hashmap_clone();


  AMRContext<dim, device_t> amr_context(forest->local_num_quadrants);

  auto flags = amr_context.m_amrflags_d;
  auto level_min = config_map.getInteger("amr", "level_min", 4);
  auto level_max = config_map.getInteger("amr", "level_max", 6);

  //
  // 2. initalize/compute an amrflags array on device (as we would do in a real application)
  //

  // get geometrical scaling factor
  const auto scaling_factor = get_scaling_factor(config_map);

  // get domain lower left corner
  const auto xyz_min = get_xyz_min<dim>(config_map);

  Kokkos::parallel_for(
    "Fill_test_data",
    Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, forest->local_num_quadrants),
    KOKKOS_LAMBDA(const int i) {
      auto key = orchard_keys_device(i);

      auto level = orchard_key_t<dim>::level(key);
      auto tree_x = orchard_key_t<dim>::template get_tree_coord<IX>(key);
      auto tree_y = orchard_key_t<dim>::template get_tree_coord<IY>(key);

      constexpr bool use_center = true;
      const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
      const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

      // create some dummy refinement criterion (geometry based)

      // reminder:
      // - we can only refine if level < level_max
      // - we can only derefine if level > level_min
      if ((level < level_max) and (tree_x == 1) and (tree_y == 1) and
          (::kalypsso::fabs(xyz[IX] - KALYPSSO_NUM(1.5)) < KALYPSSO_NUM(0.1)) and
          (::kalypsso::fabs(xyz[IY] - KALYPSSO_NUM(1.5)) < KALYPSSO_NUM(0.1)))
      {
        flags(i) = AMRContextBase::KALYPSSO_DO_REFINE;
      }
      else if ((level > level_min) and (tree_x == 1) and (tree_y == 0) and
               (::kalypsso::fabs(xyz[IX] - KALYPSSO_NUM(1.2)) < KALYPSSO_NUM(0.2)) and
               (::kalypsso::fabs(xyz[IY] - KALYPSSO_NUM(0.5)) < KALYPSSO_NUM(0.1)))
      {
        flags(i) = AMRContextBase::KALYPSSO_DO_COARSEN;
      }
      else
      {
        flags(i) = AMRContextBase::KALYPSSO_DO_NOTHING;
      }
    });

  Kokkos::deep_copy(exec_space, amr_context.m_amrflags_h, amr_context.m_amrflags_d);

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
  auto orchard_keys_device_new = mesh_map->orchard_keys_clone();

#ifdef KALYPSSO_CORE_USE_CNPY
  {
    auto orchard_keys_host_new = mesh_map->orchard_keys_host_clone();
    save_cnpy(
      Kokkos::subview(orchard_keys_host_new, std::make_pair(0, forest->local_num_quadrants)),
      "orchard_keys_after_amr",
      par_env);
  }
#endif

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
  auto userdataLeaf_new = DataArrayLeaf_t("test_data_after_amr_cycle",
                                          static_cast<size_t>(forest->local_num_quadrants),
                                          static_cast<size_t>(nbvar));
  auto userdataLeaf_new_host = Kokkos::create_mirror_view(userdataLeaf_new);

  // memory allocation for userdata on block after remapping
  auto userdataBlock_new =
    DataArrayBlock_t("test_data_block_after_amr_cycle", bSizes, nbvar, forest->local_num_quadrants);
  auto userdataBlock_face_new = FaceDataArrayBlock_t(
    "test_facedata_block_after_amr_cycle", bSizes, forest->local_num_quadrants);

  // apply data remapping - alternate version (TODO):
  // 1. find remapping index
  //    for each key in keys_new (after AMR cycle), find index "j" such that
  //    keys_old(j) less or equal than keys_new(i) less than keys_old(j+1)
  //    here doing a simple naive binary search
  // 2. apply remapping
  // WIP
  //
  // when ready, to be moved into class UserDataRemapper
  if constexpr (0)
  {
    auto local_num_octants = forest->local_num_quadrants;
    Kokkos::parallel_for(
      "DataRemap",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_num_octants),
      KOKKOS_LAMBDA(const int i) {
        const auto key_new = orchard_keys_device_new(i);

        // initialize step 1 (find remapping index, doing a simple binary search)
        int  j = i;
        int  jp1 = i;
        auto key_old = orchard_keys_host(j);

        if (key_new > orchard_keys_device(j))
          jp1 = local_num_octants - 1;
        else
          jp1 = 0;

        // TO BE CONTINUED
      });
  }

  // retrieve the new local number of octants (after AMR cycle)
  auto local_num_octants = forest->local_num_quadrants;

  // const bool on_device = true;
  // rebuild/update the new hashmap (after AMR cycle)
  // orchard keys arrays are already up to date (see line 186)
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device_new = mesh_map->hashmap_clone();
  // auto amr_hashmap_device_new = mesh_map->template create_map_device<dim>(forest, ghost_new,
  // on_device);

  KALYPSSO_INFO_ALL("after create map on device");

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

    userdataRemapper.remap_leaf_data(exec_space, userdataLeaf, userdataLeaf_new);
  } // end apply remapping

  //
  // 4.2 apply data remapping (on device) using the hash map approach - cell-centered block data
  //
  {
    KALYPSSO_INFO_ALL("after AMR local_num_quadrants = {} - remapping cell-centered data",
                      forest->local_num_quadrants);
    userdataRemapper.remap_block_data(exec_space, userdataBlock, userdataBlock_new);
  } // end apply remapping

  //
  // 4.3 apply data remapping (on device) using the hash map approach - face-centered block data
  //
  {
    KALYPSSO_INFO_ALL("after AMR local_num_quadrants = {} - remapping face-centered data",
                      forest->local_num_quadrants);
    userdataRemapper.remap_block_data(exec_space, userdataBlock_face, userdataBlock_face_new);
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
      filename, userdataLeaf_new, userdataBlock_new, config_map, amr_mesh, model);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename + "_face", userdataBlock_face_new, config_map, amr_mesh, "facedata");

    // compute divergence of the face array
    const auto divBdata =
      FaceDataArrayBlock_t::compute_divergence(userdataBlock_face_new, mesh_map->orchard_keys());

    DataWriter<dim, device_t, Hydro_t>::save_scalar(
      filename + std::string("_face_div_new"), divBdata, 0, config_map, amr_mesh);
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

    auto userdataBlock_new2 = DataArrayBlock_t(
      "test_data_block_after_partition", bSizes, nbvar, forest->local_num_quadrants);

    mesh_partitioner.repartition_userdata(userdataBlock_new, userdataBlock_new2);

    auto userdataLeaf_new2 = DataArrayLeaf_t("test_data_leaf_after_partition",
                                             static_cast<size_t>(forest->local_num_quadrants),
                                             static_cast<size_t>(nbvar));

    mesh_partitioner.repartition_userdata(userdataLeaf_new, userdataLeaf_new2);

    //
    // save data after re-partitioning
    //
    {
      std::string filename =
        dim == 2 ? "test_userdata_2d_after_partition" : "test_userdata_3d_after_partition";

      // provide mapping between variables Id and variables names
      const Hydro_t model(dim);

      DataWriter<dim, device_t, Hydro_t>::save(
        filename,
        par_env.nRanks() > 1 ? userdataLeaf_new2 : userdataLeaf_new,
        par_env.nRanks() > 1 ? userdataBlock_new2 : userdataBlock_new,
        config_map,
        amr_mesh,
        model);
    }


    // testing MPI ghosts exchange
    if constexpr (0 == 1)
    {
      // update p4est ghost after partition
      amr_mesh.reset_ghost();

      // create array of orchard keys for mirror quadrants
      // TODO: see if the following lines should be better used inside MeshGhostsExchanger
      auto mirror_keys_pair = std::move(mesh_map->template create_mirror_orchard_keys_views<dim>(
        amr_mesh.forest(), amr_mesh.ghost(), "orchard_keys_mirror_quads"));
      auto orchard_keys_mirror_host_new = mirror_keys_pair.first;
      auto orchard_keys_mirror_device_new = mirror_keys_pair.second;

      // create the main object doing MPI comm to exchange ghost data
      MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
        config_map, par_env, amr_mesh);
      mesh_ghosts_exchanger.allocate_send_recv_buffers(userdataBlock_new2.extent(0) *
                                                       userdataBlock_new2.extent(1));
    }

  } // end load balancing
#endif // KALYPSSO_CORE_USE_MPI

  const auto refine_type = config_map.getString("amr", "refine_type", "normal");
  if (refine_type == "simple")
  {
    printf("=======================================================\n");
    // display AMR keys on screen
    for (uint32_t iOct = 0; iOct < orchard_keys_host.extent(0); ++iOct)
    {
      printf("iOct=%u key=%lu\n", iOct, orchard_keys_host(iOct));
    }
  }


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
        std::cout << "Example cmdline: \"mpirun -np 1 ./test_AMRContext --ini test_AMRmesh_brick_2d.ini --refine_level 6\"\n";
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
    assertm(dim == 2 or dim == 3, "[test_AMRContext] Wrong dimension");

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
