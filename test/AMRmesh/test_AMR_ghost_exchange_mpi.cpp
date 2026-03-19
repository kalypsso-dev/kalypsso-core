// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMR_ghost_exchange_mpi.cpp
 *
 * Purpose: testing MPI communications used to fill ghost blocks.
 *
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/MeshGhostsExchanger.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>

#include <kalypsso/utils/p4est/p4est_wrapper.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>
#include <string>

namespace kalypsso
{

// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{

  using Hydro_t = core::models::Hydro;
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  InitialAMRSetup<dim, device_t, InitFunc2> initial_amr_setup(par_env, config_map, InitFunc2{});
  const auto                                f = InitFunc2{};

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

  // mirror keys array must be up to date for MeshGhostExchange to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());

  // auto [userdata_leaf, userdata_block] =
  // initial_amr_setup.setup_initial_data(orchard_keys_device);
  auto userdata_leaf = initial_amr_setup.setup_initial_data_leaf(orchard_keys_device);
  auto userdata_block = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);
  auto userdata_block_face =
    initial_amr_setup.setup_initial_data_block_face(orchard_keys_device, false);

  //
  // save data before MPI ghost exchange
  //
  {
    std::string filename =
      dim == 2 ? "test_AMR_ghost_exchange_mpi_before_2d" : "test_AMR_ghost_exchange_mpi_before_3d";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_leaf, userdata_block, config_map, amr_mesh, model);
  }

  //
  // here comes the interesting part : perform MPI ghosts exchange
  //

  //
  // 1. store the UnorderedMap before MPI ghost exchange
  //

  // make sure p4est ghost is up to date (should be ok, since initial AMR setup already initialized
  // ghost)
  // amr_mesh.reset_ghost();
  auto ghost = amr_mesh.ghost();

  auto userdata_leaf_host = Kokkos::create_mirror_view(userdata_leaf);

  auto userdata_block_host = DataArrayBlock_t::create_host_mirror_view(userdata_block);

  // compute amr keys map
  const bool on_device = true;

  // retrieve hash map of orchard keys
  mesh_map->update_hashmap(on_device);
  // auto amrkeys_hashmap = mesh_map->hashmap();

  //
  // testing MPI ghosts exchange
  //
  {

    // create array of orchard keys for mirror quadrants
    // TODO: see if the following lines should be better used inside MeshGhostsExchanger
    // auto mirror_keys_pair = mesh_map->template create_mirror_orchard_keys_views<dim>(
    //  amr_mesh.forest(), amr_mesh.ghost(), "orchard_keys_mirror_quads");
    // auto orchard_keys_mirror_device = mirror_keys_pair.second;

    // create the main object doing MPI comm to exchange ghost data
    MeshGhostsExchanger<dim, real_t, device_t> mesh_ghosts_exchanger(
      config_map, par_env, amr_mesh, *mesh_map);
    MPI_Barrier(par_env.mpi_comm());

    mesh_ghosts_exchanger.exchange(userdata_leaf);
    Kokkos::deep_copy(userdata_leaf_host, userdata_leaf);

    mesh_ghosts_exchanger.exchange(userdata_block);
    Kokkos::deep_copy(userdata_block_host.logical_view(), userdata_block.logical_view());

    mesh_ghosts_exchanger.exchange(userdata_block_face);
    auto userdata_block_face_flat_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, userdata_block_face.logical_view());

    auto orchard_keys_mirror_host = mesh_map->mirror_orchard_keys_host();

    // get geometrical scaling factor
    const auto scaling_factor = get_scaling_factor(config_map);

    // get domain lower left corner
    const auto xyz_min = get_xyz_min<dim>(config_map);

    //
    // HERE is the MAIN test
    //

    // in the following test, the pivot rank prints its mirror quadrants data (to be sent), and all
    // other MPI processes print their ghost data (received from the pivot rank)
    //
    // a more exhaustive test should use all possible values for the pivot rank, i.e. from 0 to
    // MPI_Comm_size - 1.
    int pivot_rank = 1;

    // ====================================================================
    //
    // LEAF data
    //
    // ====================================================================

    // cross-checking that mirrors quadrants in rank "pivot_rank" are also ghosts in
    // other mpi ranks
    for (int irank = 0; irank < par_env.size(); ++irank)
    {
      if (irank == par_env.rank())
      {
        if (par_env.rank() == pivot_rank)
        {
          constexpr bool use_center = false;
          for (int iproc = 0; iproc < par_env.size(); ++iproc)
          {
            auto first_mirror = ghost->mirror_proc_offsets[iproc];
            auto num_mirrors =
              ghost->mirror_proc_offsets[iproc + 1] - ghost->mirror_proc_offsets[iproc];
            for (int32_t iOct = first_mirror; iOct < first_mirror + num_mirrors; ++iOct)
            {
              const auto key = orchard_keys_mirror_host(iOct);
              const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
              const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

              if constexpr (dim == 2)
              {
                KALYPSSO_INFO("[rank={}] mirror to proc={} imirror={}, key={} x={} y=%{}\n",
                              par_env.rank(),
                              iproc,
                              iOct,
                              key,
                              xyz[IX],
                              xyz[IY]);
              }
              else if constexpr (dim == 3)
              {
                KALYPSSO_INFO("[rank={}] mirror to proc={} imirror={}, key={} x={} y={} z={}\n",
                              par_env.rank(),
                              iproc,
                              iOct,
                              key,
                              xyz[IX],
                              xyz[IY],
                              xyz[IZ]);
              }
            }
          }
        }
        else
        {
          constexpr bool use_center = false;
          auto first_ghost = forest->local_num_quadrants + ghost->proc_offsets[pivot_rank];
          auto num_ghosts = ghost->proc_offsets[pivot_rank + 1] - ghost->proc_offsets[pivot_rank];

          for (int32_t iOct = first_ghost; iOct < first_ghost + num_ghosts; ++iOct)
          {
            const auto key = orchard_keys_host(iOct);
            const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
            const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

            if constexpr (dim == 2)
            {
              KALYPSSO_INFO("[rank={}] ghost i={}, key={} x={} y={} data_leaf={}\n",
                            par_env.rank(),
                            iOct,
                            key,
                            xyz[IX],
                            xyz[IY],
                            userdata_leaf_host(iOct, 1));
            }
            else if constexpr (dim == 3)
            {
              KALYPSSO_INFO("[rank={}] ghost i={}, key={} x={} y={} z={} data_leaf={}\n",
                            par_env.rank(),
                            iOct,
                            key,
                            xyz[IX],
                            xyz[IY],
                            xyz[IZ],
                            userdata_leaf_host(iOct, 1));
            }
          }
        }
      } // end if(irank == par_env.rank())
      MPI_Barrier(par_env.mpi_comm());
    } // end for irank

    // ====================================================================
    //
    // cell-center block data
    //
    // ====================================================================

    MPI_Barrier(par_env.mpi_comm());
    KALYPSSO_INFO("\n\n");

    for (int irank = 0; irank < par_env.size(); ++irank)
    {
      if (irank == par_env.rank())
      {
        if (par_env.rank() == pivot_rank)
        {
          constexpr bool use_center = false;
          for (int iproc = 0; iproc < par_env.size(); ++iproc)
          {
            auto first_mirror = ghost->mirror_proc_offsets[iproc];
            auto num_mirrors =
              ghost->mirror_proc_offsets[iproc + 1] - ghost->mirror_proc_offsets[iproc];
            for (int32_t iOct = first_mirror; iOct < first_mirror + num_mirrors; ++iOct)
            {
              const auto key = orchard_keys_mirror_host(iOct);
              const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
              const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

              if constexpr (dim == 2)
              {
                KALYPSSO_INFO("[rank={}] mirror to proc={} imirror={}, key={} x={} y={} - block\n",
                              par_env.rank(),
                              iproc,
                              iOct,
                              key,
                              xyz[IX],
                              xyz[IY]);
              }
              else if constexpr (dim == 3)
              {
                KALYPSSO_INFO(
                  "[rank={}] mirror to proc={} imirror={}, key={} x={} y={} z={} - block\n",
                  par_env.rank(),
                  iproc,
                  iOct,
                  key,
                  xyz[IX],
                  xyz[IY],
                  xyz[IZ]);
              }
            }
          }
        }
        else
        {
          constexpr bool use_center = false;
          auto first_ghost = forest->local_num_quadrants + ghost->proc_offsets[pivot_rank];
          auto num_ghosts = ghost->proc_offsets[pivot_rank + 1] - ghost->proc_offsets[pivot_rank];
          const auto block_sizes = initial_amr_setup.block_sizes();

          for (int32_t iOct = first_ghost; iOct < first_ghost + num_ghosts; ++iOct)
          {
            const auto key = orchard_keys_host(iOct);
            const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
            const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);
            const auto level = orchard_key_t<dim>::level(key);

            const int32_t ivar = 1;
            const int32_t icell = 6;
            const auto    iCoord = icell_to_icoord<dim>(icell, block_sizes[IX]);
            const auto    xyz_cell_vertex =
              compute_cell_coordinates<dim>(level, xyz_vertex, iCoord, block_sizes);
            const auto xyz_cell =
              vertex_coord_to_real_space<dim>(xyz_cell_vertex, scaling_factor, xyz_min);

            // make sure data recv in ghost block are the same as the analytical expected value
            // (provided by functor f)
            if constexpr (dim == 2)
            {
              KALYPSSO_INFO(
                "[rank={}] ghost i={}, key={} x={} y={} data_block_recv={} data_block_true={} "
                "- diff={}\n",
                par_env.rank(),
                iOct,
                key,
                xyz[IX],
                xyz[IY],
                userdata_block_host(icell, ivar, iOct),
                f(xyz_cell[IX], xyz_cell[IY], ivar),
                userdata_block_host(icell, ivar, iOct) - f(xyz_cell[IX], xyz_cell[IY], ivar));
            }
            else if constexpr (dim == 3)
            {
              KALYPSSO_INFO("[rank={}] ghost i={}, key={} x={} y={} z={} data_block_recv={} "
                            "data_block_true={} "
                            "- diff={}\n",
                            par_env.rank(),
                            iOct,
                            key,
                            xyz[IX],
                            xyz[IY],
                            xyz[IZ],
                            userdata_block_host(icell, ivar, iOct),
                            f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar),
                            userdata_block_host(icell, ivar, iOct) -
                              f(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar));
            }
          }
        }
      } // end if(irank == par_env.rank())
      MPI_Barrier(par_env.mpi_comm());
    } // end for irank

    // ====================================================================
    //
    // face-center block data
    //
    // ====================================================================

    MPI_Barrier(par_env.mpi_comm());
    KALYPSSO_INFO("\n\n");

    for (int irank = 0; irank < par_env.size(); ++irank)
    {
      if (irank == par_env.rank())
      {
        if (par_env.rank() == pivot_rank)
        {
          constexpr bool use_center = false;
          for (int iproc = 0; iproc < par_env.size(); ++iproc)
          {
            auto first_mirror = ghost->mirror_proc_offsets[iproc];
            auto num_mirrors =
              ghost->mirror_proc_offsets[iproc + 1] - ghost->mirror_proc_offsets[iproc];
            for (int32_t iOct = first_mirror; iOct < first_mirror + num_mirrors; ++iOct)
            {
              const auto key = orchard_keys_mirror_host(iOct);
              const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
              const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor, xyz_min);

              if constexpr (dim == 2)
              {
                KALYPSSO_INFO(
                  "[rank={}] mirror to proc={} imirror={}, key={} x={} y={} - block face\n",
                  par_env.rank(),
                  iproc,
                  iOct,
                  key,
                  xyz[IX],
                  xyz[IY]);
              }
              else if constexpr (dim == 3)
              {
                KALYPSSO_INFO(
                  "[rank={}] mirror to proc={} imirror={}, key={} x={} y={} z={} - block face\n",
                  par_env.rank(),
                  iproc,
                  iOct,
                  key,
                  xyz[IX],
                  xyz[IY],
                  xyz[IZ]);
              }
            }
          }
        }
        else
        {
          // constexpr bool use_center = false;
          auto first_ghost = forest->local_num_quadrants + ghost->proc_offsets[pivot_rank];
          auto num_ghosts = ghost->proc_offsets[pivot_rank + 1] - ghost->proc_offsets[pivot_rank];
          const auto block_sizes = initial_amr_setup.block_sizes();
          const auto num_elts_per_octant = userdata_block_face.num_elements_per_octant();

          for (int32_t iOct = first_ghost; iOct < first_ghost + num_ghosts; ++iOct)
          {
            const auto key = orchard_keys_host(iOct);
            // const auto xyz_vertex = orchard_key_to_vertex_coord<dim>(key, use_center);
            // const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, scaling_factor,
            // xyz_min);

            // face flat index is in range [0, userdata_block_face.num_elements_per_octants-1]
            const int32_t face_flat_index = 6;

            const auto face_indexes = face_flat_index_unravel<dim>(face_flat_index,
                                                                   block_sizes,
                                                                   userdata_block_face.offsets(),
                                                                   userdata_block_face.shift());

            const auto xyz = orchard_key_to_facecenter_real_space<dim>(
              key, face_indexes, block_sizes[IX], scaling_factor, xyz_min);

            const auto & ivar = face_indexes[dim];

            // make sure data recv in ghost block are the same as the analytical expected value
            // (provided by functor f)
            if constexpr (dim == 2)
            {
              KALYPSSO_INFO(
                "[rank={}] ghost i={}, key={} x={} y={} data_block_face_recv={} "
                "data_block_face_true={} "
                "- diff={}\n",
                par_env.rank(),
                iOct,
                key,
                xyz[IX],
                xyz[IY],
                userdata_block_face_flat_host(face_flat_index + num_elts_per_octant * iOct),
                f(xyz[IX], xyz[IY], ivar),
                userdata_block_face_flat_host(face_flat_index + num_elts_per_octant * iOct) -
                  f(xyz[IX], xyz[IY], ivar));
            }
            else if constexpr (dim == 3)
            {
              KALYPSSO_INFO(
                "[rank={}] ghost i={}, key={} x={} y={} z={} data_block_face_recv={} "
                "data_block_face_true={} "
                "- diff={}\n",
                par_env.rank(),
                iOct,
                key,
                xyz[IX],
                xyz[IY],
                xyz[IZ],
                userdata_block_face_flat_host(face_flat_index + num_elts_per_octant * iOct),
                f(xyz[IX], xyz[IY], xyz[IZ], ivar),
                userdata_block_face_flat_host(face_flat_index + num_elts_per_octant * iOct) -
                  f(xyz[IX], xyz[IY], xyz[IZ], ivar));
            }
          }
        }
      } // end if(irank == par_env.rank())
      MPI_Barrier(par_env.mpi_comm());
    } // end for irank

  } // end testing MPI ghosts exchange

  MPI_Barrier(par_env.mpi_comm());

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
        std::cout << "Example cmdline: \"mpirun -np 4 ./test_AMR_ghost_exchange_mpi --ini test_AMRmesh_brick_2d_ghost_exchange_mpi.ini --refine_level 3\"\n";
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
      config_filename = threed ? "./test_AMRmesh_brick_3d_ghost_exchange_mpi.ini"
                               : "./test_AMRmesh_brick_2d_ghost_exchange_mpi.ini";

    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(config_filename);

    // check if input file is valid, i.e. parameter run/dimension is 2 or 3
    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[test_AMR_ghost_exchange_mpi] Wrong dimension");

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
