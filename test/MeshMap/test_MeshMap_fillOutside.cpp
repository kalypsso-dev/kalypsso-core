// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_MeshMap_fillOutside.cpp
 *
 * purpose: just illustrate usage of functor class FillOutsideCellFunctor
 */

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/MeshMap.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/cmdline_utils.h>

#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/p4est/connectivity.h>
#include <kalypsso/utils/p4est/geometry.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>
#include <test_common/FillOutside.h>

#include <cstdlib>

namespace kalypsso
{

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, int argc, char * argv[])
{
  // check if user passed a custom ini filename
  std::string config_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

  // provide a default config filename (that exists)
  if (config_filename.size() == 0)
    config_filename = dim == 2 ? "./test_MeshMap_brick_2d.ini" : "./test_MeshMap_brick_3d.ini";

  ConfigMap config_map = broadcast_parameters(config_filename);

  if (par_env.rank() == 0)
    printf("================================================\n");


  InitialAMRSetup<dim, device_t, InitFunc1> initial_amr_setup(par_env, config_map, InitFunc1{});

  auto level_min = config_map.getInteger("amr", "level_min", 2);
  auto level_max = config_map.getInteger("amr", "level_max", 2);
  auto no_refine = (level_min == level_max);

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto &     amr_mesh = initial_amr_setup.mesh();
  auto       forest = amr_mesh.forest();
  auto       mesh_map = initial_amr_setup.mesh_map();
  const auto block_sizes = initial_amr_setup.block_sizes();
  const auto brick_sizes = initial_amr_setup.brick_sizes();
  // const auto is_brick_periodic = initial_amr_setup.is_brick_periodic();
  const auto amr_mesh_info = initial_amr_setup.amr_mesh_info();
  if (par_env.rank() == 0)
  {
    amr_mesh_info.print();
  }


  // initialize ghost
  amr_mesh.reset_ghost();
  auto ghost = amr_mesh.ghost();

  mesh_map->compute_outside_quad_info(forest, ghost);
  mesh_map->update_orchard_keys(forest, ghost);
  mesh_map->update_hashmap(forest, ghost);

  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

  // extract userdata
  auto userdata_leaf_d = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdata_block_d = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);

  auto amr_hashmap_device = mesh_map->hashmap();

  using bc_array_t = BorderConditionsConfig<test::BC_HYDRO>::bc_array_t<dim>;

  bc_array_t bc_types = [=]() {
    if constexpr (dim == 2)
    {
      // XMIN, XMAX, YMIN, YMAX

      // clang-format off
      // return bc_array_t{ test::BC_HYDRO::ZERO_GRADIENT,
      //                    test::BC_HYDRO::ZERO_GRADIENT,
      //                    test::BC_HYDRO::ZERO_GRADIENT,
      //                    test::BC_HYDRO::ZERO_GRADIENT };

      return bc_array_t{ test::BC_HYDRO::WALL,
                         test::BC_HYDRO::WALL,
                         test::BC_HYDRO::WALL,
                         test::BC_HYDRO::WALL };
      // clang-format on
    }
    else if constexpr (dim == 3)
    {
      // XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
      return bc_array_t{ test::BC_HYDRO::ZERO_GRADIENT, test::BC_HYDRO::ZERO_GRADIENT,
                         test::BC_HYDRO::ZERO_GRADIENT, test::BC_HYDRO::ZERO_GRADIENT,
                         test::BC_HYDRO::ZERO_GRADIENT, test::BC_HYDRO::ZERO_GRADIENT };
    }
  }();

  // fill outside userdata
  test::FillOutsideCellFunctor<dim, device_t>::apply(config_map,
                                                     amr_hashmap_device,
                                                     orchard_keys_device,
                                                     amr_mesh_info,
                                                     userdata_block_d,
                                                     block_sizes,
                                                     brick_sizes,
                                                     mesh_map->is_brick_periodic(),
                                                     bc_types);

  // save data inside / outside
#ifdef KALYPSSO_CORE_USE_HDF5
  {
    // // create host mirrors
    // auto userdata_leaf_h =
    //   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, userdata_leaf_d);
    // auto userdata_block_h =
    //   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, userdata_block_d);

    std::string filename = dim == 2 ? "test_MeshMap_fillOutside_2d" : "test_MeshMap_fillOutside_3d";

    // provide mapping between variables Id and variables names
    using Hydro_t = core::models::Hydro;
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(filename,
                                             userdata_leaf_d,
                                             userdata_block_d,
                                             config_map,
                                             par_env,
                                             mesh_map,
                                             false, // inside
                                             model);

    DataWriter<dim, device_t, Hydro_t>::save(filename,
                                             userdata_leaf_d,
                                             userdata_block_d,
                                             config_map,
                                             par_env,
                                             mesh_map,
                                             true, // outside
                                             model);
  }
#else
  std::cout << "HDF5 output unavailable.\n";
#endif

} // run_test

} // namespace kalypsso

// =============================================================================
// =============================================================================
// =============================================================================
int
main(int argc, char * argv[])
{

  {
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_MeshMap_fillOutside --ini test_MeshMap_brick_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // assign static global variables based on configuration
    kalypsso::InitialAMRSetupBase::refine_level =
      kalypsso::cmdline_arg_exists(argv, argv + argc, "--refine_level")
        ? kalypsso::cmdline_get_integer(argv, argv + argc, "--refine_level")
        : 6;

    if (par_env.rank() == 0)
      printf("Using refine_level=%d\n", kalypsso::InitialAMRSetupBase::refine_level);

    // parse command line : 2d or 3d ?
    bool run_3d = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

    run_3d ? kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, argc, argv)
           : kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, argc, argv);
  }

  return 0;
} // main
