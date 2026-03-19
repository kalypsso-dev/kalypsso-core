// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh_fill_block_ghost_cells.cpp
 *
 * Purpose: test and validate code used to fill ghost cells around blocks (leaf of octree).
 *
 * Input array is non-ghosted
 * Output array is ghosted
 *
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/core/init_func.h>

#include <kalypsso/core/FillBlockGhostCells.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>
#include <test_common/FillOutside.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>
#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>

namespace kalypsso
{

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t, typename InitFunc>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{
  using exec_space = typename device_t::execution_space;

  using Hydro_t = core::models::Hydro;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  InitialAMRSetup<dim, device_t, InitFunc> initial_amr_setup(par_env, config_map, InitFunc{});

  auto level_min = config_map.getInteger("amr", "level_min", 2);
  auto level_max = config_map.getInteger("amr", "level_max", 2);
  auto no_refine = (level_min == level_max);

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   mesh_map = initial_amr_setup.mesh_map();
  auto   brick_sizes = initial_amr_setup.brick_sizes();
  auto   is_brick_periodic = initial_amr_setup.is_brick_periodic();
  auto   amr_mesh_info = initial_amr_setup.amr_mesh_info();

  // initialize ghost
  amr_mesh.reset_ghost();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host();
  auto orchard_keys_device = mesh_map->orchard_keys();

#ifdef KALYPSSO_CORE_USE_MPI
  // mirror keys array must be up to date for MeshGhostExchanger to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());
#endif // KALYPSSO_CORE_USE_MPI

  // rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device = mesh_map->hashmap();

  // auto [userdata_leaf, userdata_block] =
  // initial_amr_setup.setup_initial_data(orchard_keys_device);
  auto userdata_leaf = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdata_block = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);

  userdata_block.resize(amr_mesh_info.local_num_quadrants_total());

  // auto userdata_block_host = Kokkos::create_mirror_view(userdata_block);
  auto userdata_block_host = DataArrayBlock_t::create_host_mirror_view(userdata_block);

  auto userdata_ghosted_block =
    initial_amr_setup.setup_initial_data_ghosted_block(orchard_keys_device, true);

  userdata_ghosted_block.resize(amr_mesh_info.local_num_quadrants_total());

  // reset (for test)
  Kokkos::deep_copy(userdata_ghosted_block.flat_view(), 0.0);

  //
  // save data before filling block ghosts
  //
  {
    std::string filename = dim == 2 ? "test_AMR_fill_block_noghosts_2d_before"
                                    : "test_AMR_fill_block_noghosts_3d_before";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(filename, userdata_block, config_map, amr_mesh, model);
  }

#ifdef KALYPSSO_CORE_USE_MPI
  //
  // make sure MPI ghosts are OK
  //

  // create the main object doing MPI comm to exchange ghost data
  MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
    config_map, par_env, amr_mesh, *mesh_map);
  MPI_Barrier(par_env.mpi_comm());

  // auto ghosted_data = userdata_ghosted_block.view();

  mesh_ghosts_exchanger.exchange(userdata_block);

  // auto ghosted_data_host = Kokkos::create_mirror_view(ghosted_data);
  // Kokkos::deep_copy(ghosted_data_host, ghosted_data);
#endif // KALYPSSO_CORE_USE_MPI

  //
  // Make sure outside cells are ok (this non-op if user set the mesh as periodic)
  //
  {
    using bc_array_t = BorderConditionsConfig<test::BC_HYDRO>::bc_array_t<dim>;

    bc_array_t bc_types = [=]() {
      if constexpr (dim == 2)
      {
        // clang-format off
        return bc_array_t{
          test::BC_HYDRO::ANALYTICAL,
          test::BC_HYDRO::ANALYTICAL,
          test::BC_HYDRO::ANALYTICAL,
          test::BC_HYDRO::ANALYTICAL
        };
        // clang-format on
      }
      else if constexpr (dim == 3)
      {
        return bc_array_t{ test::BC_HYDRO::ANALYTICAL, test::BC_HYDRO::ANALYTICAL,
                           test::BC_HYDRO::ANALYTICAL, test::BC_HYDRO::ANALYTICAL,
                           test::BC_HYDRO::ANALYTICAL, test::BC_HYDRO::ANALYTICAL };
      }
    }();

    // fill outside userdata
    test::FillOutsideCellFunctor<dim, device_t, InitFunc>::apply(config_map,
                                                                 amr_hashmap_device,
                                                                 orchard_keys_device,
                                                                 amr_mesh_info,
                                                                 userdata_block,
                                                                 initial_amr_setup.block_sizes(),
                                                                 brick_sizes,
                                                                 mesh_map->is_brick_periodic(),
                                                                 bc_types,
                                                                 InitFunc{});
  }

  //
  // filling ghost blocks (piecewise)
  //
  {
    const auto num_cells = userdata_ghosted_block.num_cells();
    const auto num_vars = userdata_ghosted_block.num_vars();

    const auto nbOcts = amr_mesh_info.local_num_quadrants();

    // process octant in a picewise manner
    const int32_t num_octants_per_group = 32;

    const int32_t nbGroups = (nbOcts + num_octants_per_group - 1) / num_octants_per_group;

    auto userdata_ghosted_block_range =
      DataArrayGhostedBlock<dim, real_t, device_t>(userdata_ghosted_block.block_size(),
                                                   userdata_ghosted_block.ghosted_block_size(),
                                                   userdata_ghosted_block.shift(),
                                                   "userdata_ghosted_block_range",
                                                   userdata_ghosted_block.num_vars(),
                                                   num_octants_per_group);

    for (int32_t iGroup = 0; iGroup < nbGroups; ++iGroup)
    {
      const auto iOct_begin = iGroup * num_octants_per_group;
      const auto num_octants_to_process =
        (iGroup == (nbGroups - 1)) ? nbOcts - iOct_begin : num_octants_per_group;

      FillBlockGhostCellsFunctor<dim, device_t>::apply(config_map,
                                                       amr_hashmap_device,
                                                       orchard_keys_device,
                                                       nbOcts,
                                                       iOct_begin,
                                                       num_octants_to_process,
                                                       userdata_block,
                                                       userdata_ghosted_block_range,
                                                       brick_sizes,
                                                       is_brick_periodic);

      // copy back results
      {
        // source view
        auto data_range_in =
          std::pair<std::size_t, std::size_t>(0, num_cells * num_vars * num_octants_to_process);
        auto userdata_ghosted_block_range_view =
          Kokkos::subview(userdata_ghosted_block_range.flat_view(), data_range_in);

        // destination view
        auto data_range_out = std::pair<std::size_t, std::size_t>(
          num_cells * num_vars * iOct_begin,
          num_cells * num_vars * (iOct_begin + num_octants_to_process));
        auto userdata_ghosted_block_subview =
          Kokkos::subview(userdata_ghosted_block.flat_view(), data_range_out);

        // copy
        Kokkos::deep_copy(userdata_ghosted_block_subview, userdata_ghosted_block_range_view);
      }
    }
  }

  {
    std::string filename =
      dim == 2 ? "test_AMR_fill_block_ghosts_2d_after" : "test_AMR_fill_block_ghosts_3d_after";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_ghosted_block, config_map, amr_mesh, true, model);
  }

  // create data array with ghost block filled with analytical values
  auto userdata_ghosted_block_true =
    initial_amr_setup.setup_initial_data_ghosted_block(orchard_keys_device, true);

  // perform comparison
  auto diff = initial_amr_setup.compute_diff_ghosted_block(
    orchard_keys_device, userdata_ghosted_block, userdata_ghosted_block_true);

  {
    std::string filename =
      dim == 2 ? "test_AMR_fill_block_ghosts_2d_diff" : "test_AMR_fill_block_ghosts_3d_diff";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(filename, diff, config_map, amr_mesh, true, model);
  }

  //
  // perform a reduce to check everything is ok (this is a unit test)
  //
  real_t              l2_square_diff = 0;
  Kokkos::Sum<real_t> reducer_diff(l2_square_diff);
  auto                diff_v = diff.data();
  auto                userdata_ghosted_block_v = userdata_ghosted_block.data();
  auto                userdata_ghosted_block_true_v = userdata_ghosted_block_true.data();

  Kokkos::parallel_reduce(
    "Compute_l2_square_diff",
    Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
      { 0, 0, 0 },
      { diff_v.num_cells(),
        diff_v.num_vars(),
        //(int32_t)diff_v.num_quadrants()
        amr_mesh_info.local_num_quadrants() + 0 * amr_mesh_info.local_num_ghosts() }),
    KOKKOS_LAMBDA(const int32_t i, const int32_t j, const int32_t k, real_t & local_sum) {
      local_sum += diff_v(i, j, k) * diff_v(i, j, k);
      // if (diff_v(i, j, k) * diff_v(i, j, k) > 0)
      // {
      //   KALYPSSO_INFO("error at ({},{},{}) {} | {} {}\n",
      //          i,
      //          j,
      //          k,
      //          diff_v(i, j, k),
      //          userdata_ghosted_block_v(i, j, k),
      //          userdata_ghosted_block_true_v(i, j, k));
      // }
    },
    reducer_diff);

  real_t              l2_square_true = 0;
  Kokkos::Sum<real_t> reducer_true(l2_square_true);
  // auto                userdata_ghosted_block_v = userdata_ghosted_block.view();

  Kokkos::parallel_reduce(
    "Compute_l2_square_true",
    Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
      { 0, 0, 0 },
      { userdata_ghosted_block_v.num_cells(),
        userdata_ghosted_block_v.num_vars(),
        /*(int32_t)userdata_ghosted_block_v.num_quadrants()*/
        amr_mesh_info.local_num_quadrants() }),
    KOKKOS_LAMBDA(const int32_t i, const int32_t j, const int32_t k, real_t & local_sum) {
      local_sum += userdata_ghosted_block_v(i, j, k) * userdata_ghosted_block_v(i, j, k);
    },
    reducer_true);

  KALYPSSO_INFO("[MPI rank={}] Relative L2 norm is : {}\n",
                par_env.rank(),
                sqrt(l2_square_diff / l2_square_true));

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
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_AMRmesh_fill_block_ghost_cells --ini test_AMRmesh_fill_ghost_brick_2d.ini --refine_level 6\"\n";
        // clang-format on
      }
      return 0;
    }

    // parse command line : 2d or 3d ?
    bool use_3d = kalypsso::cmdline_arg_exists(argv, argv + argc, "--3d");

    // check if user passed a custom ini filename
    std::string config_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    // provide a default config filename (that exists)
    if (config_filename.size() == 0)
      config_filename = use_3d ? "./test_AMRmesh_fill_ghost_brick_3d.ini"
                               : "./test_AMRmesh_fill_ghost_brick_2d.ini";

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_AMRmesh_fill_block_ghost_cells] Wrong dimension");

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

    using InitFunc = kalypsso::InitFunc2;

    if (dim == 2)
    {
      kalypsso::run_test<2, kalypsso::DefaultDevice, InitFunc>(par_env, config_map);
    }
    else if (dim == 3)
    {
      kalypsso::run_test<3, kalypsso::DefaultDevice, InitFunc>(par_env, config_map);
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
