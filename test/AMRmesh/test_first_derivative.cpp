// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_first_derivative.cpp
 *
 * Purpose:
 * test and validate code used to compute 1st derivative on 3 or 5 points stencil.
 *
 */

#include <kalypsso/core/FirstOrderDerivativeFiniteDifference.h>

#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

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
  // using exec_space = typename device_t::execution_space;

  // using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  //  using DataArrayBlockHost_t = DataArrayBlockHost<real_t, device_t>;

  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  using Hydro_t = core::models::Hydro;

  InitialAMRSetup<dim, device_t, InitFuncParabola> initial_amr_setup(
    par_env, config_map, InitFuncParabola{});
  // InitialAMRSetup<dim, device_t, InitFuncSineWave> initial_amr_setup(
  //   par_env, config_map, InitFuncSineWave{});

  const auto level_min = config_map.getInteger("amr", "level_min", 2);
  const auto level_max = config_map.getInteger("amr", "level_max", 2);
  const auto no_refine = (level_min == level_max);

  // setup initial p4est mesh with some test pseudo random refined/coarsen cells pattern
  initial_amr_setup.setup_initial_mesh(no_refine);

  auto & amr_mesh = initial_amr_setup.mesh();
  auto   mesh_map = initial_amr_setup.mesh_map();
  auto   amr_mesh_info = initial_amr_setup.amr_mesh_info();

  // retrieve amr keys
  mesh_map->update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

#ifdef KALYPSSO_CORE_USE_MPI
  // mirror keys array must be up to date for MeshGhostExchanger to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());
#endif // KALYPSSO_CORE_USE_MPI

  //  rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);

  auto userdata_in = initial_amr_setup.setup_initial_data_ghosted_block(orchard_keys_device, false);
  userdata_in.resize(amr_mesh_info.local_num_quadrants_total());

  // auto userdata_in_host = DataArrayGhostedBlock_t::create_host_mirror_view(userdata_in);

  auto userdata_out = DataArrayGhostedBlock_t(userdata_in.block_size(),
                                              userdata_in.block_size() + 2 * 1,
                                              get_shift<dim>(-1),
                                              "userdata_out",
                                              userdata_in.num_vars(),
                                              userdata_in.num_quadrants());
  // Kokkos::deep_copy(userdata_out.data().storage(), -1.0);

  //
  // save data before stencil computation
  //
  // {
  //   std::string filename = dim == 2 ? "test_AMR_stencil_2d_before" :
  //   "test_AMR_stencil_3d_before";

  //   DataWriter<dim, device_t, Hydro_t>::save(
  //     filename, userdata_in, config_map, amr_mesh);
  // }

#ifdef KALYPSSO_CORE_USE_MPI
  //
  // make sure MPI ghosts are OK
  //

  // create the main object doing MPI comm to exchange ghost data
  // MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
  //   config_map, par_env, amr_mesh, *mesh_map);
  // MPI_Barrier(par_env.mpi_comm());

  // mesh_ghosts_exchanger.exchange(userdata_in);
#endif // KALYPSSO_CORE_USE_MPI

  //
  // compute first derivative
  //
  {
    const auto stencil_length = core::FIRST_DERIVATIVE_STENCIL::THREE_POINTS;
    // const auto stencil_length = core::FIRST_DERIVATIVE_STENCIL::FIVE_POINTS;
    // const auto stencil_length = core::FIRST_DERIVATIVE_STENCIL::SEVEN_POINTS;

    const auto nbOcts = amr_mesh_info.local_num_quadrants();

    core::FirstOrderDerivativeFiniteDifference<dim, device_t>::first_derivative(config_map,
                                                                                orchard_keys_device,
                                                                                0,
                                                                                nbOcts,
                                                                                userdata_in,
                                                                                0,
                                                                                userdata_out,
                                                                                0,
                                                                                IX,
                                                                                stencil_length);

    core::FirstOrderDerivativeFiniteDifference<dim, device_t>::first_derivative(config_map,
                                                                                orchard_keys_device,
                                                                                0,
                                                                                nbOcts,
                                                                                userdata_in,
                                                                                0,
                                                                                userdata_out,
                                                                                1,
                                                                                IY,
                                                                                stencil_length);

    {
      std::string filename = dim == 2 ? "test_AMR_stencil_2d_after" : "test_AMR_stencil_3d_after";

      // provide mapping between variables Id and variables names
      const Hydro_t model(dim);

      DataWriter<dim, device_t, Hydro_t>::save(
        filename, userdata_out, config_map, amr_mesh, true, model);
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
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_first_derivative --ini test_AMRmesh_stencil_2d.ini --refine_level 6\"\n";
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
      config_filename = use_3d ? "./test_AMRmesh_stencil_3d.ini" : "./test_AMRmesh_stencil_2d.ini";

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_first_derivative] Wrong dimension");

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
