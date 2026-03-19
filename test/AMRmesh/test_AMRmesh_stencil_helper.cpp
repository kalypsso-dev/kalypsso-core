// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file test_AMRmesh_stencil_help.cpp
 *
 *
 */

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/io_utils.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>
#include <kalypsso/core/real_type.h>

#include <kalypsso/core/StencilHelper.h>

#include <test_common/InitialAMRSetup.h>
#include <test_common/DataWriter.h>

#include <kalypsso/utils/config/ConfigMap.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <cstdlib>
#include <list>

namespace kalypsso
{
// =============================================================================
// =============================================================================
template <size_t dim, typename device_t>
void
run_test(const ParallelEnv & par_env, const ConfigMap & config_map)
{
  using StencilHelper_t = StencilHelper<dim, device_t>;
  using CellLocation_t = CellLocation<dim>;

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
  auto orchard_keys_host = mesh_map->orchard_keys_host_clone();
  auto orchard_keys_device = mesh_map->orchard_keys_clone();

  // mirror keys array must be up to date for MeshGhostExchange to be functional
  mesh_map->update_mirror_orchard_keys(amr_mesh.ghost());

  //  rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device = mesh_map->hashmap();
  mesh_map->update_conformal_status();
  auto conformal_status = mesh_map->conformal_status();

  {
    auto amr_mesh_info = initial_amr_setup.amr_mesh_info();
    auto userdata_block_in = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);
    userdata_block_in.resize(amr_mesh_info.local_num_quadrants_total());

    std::string filename = dim == 2 ? "test_AMR_stencil_helper_2d" : "test_AMR_stencil_helper_3d";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_block_in, config_map, amr_mesh, model);
  }

  const auto block_sizes_emf = block_sizes + 1;

  auto stencil_helper = StencilHelper_t(
    amr_hashmap_device, orchard_keys_device, block_sizes, brick_sizes, is_brick_periodic);


  // const auto iOct = 1235;
  // const auto iOct = 1246;
  // const auto iOct = 1248;
  // const auto iOct = 1264;
  std::list<uint32_t> octs = { 1235, 1246, 1248, 1264 };

  for (const auto & iOct : octs)
  {
    const auto face_xmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmin(conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    // const auto face_xmin_neighbor_is_finer =
    //   conformal_face_status_t<dim>::face_xmin(conformal_status(iOct)) ==
    //   conformal_neighbor_status::NEIGHBOR_IS_FINER;

    for (int32_t cell_index = 0; cell_index < Kokkos::dim_prod(block_sizes_emf); ++cell_index)
    {
      auto const   coords_face = cellindex_to_coord<dim>(cell_index, block_sizes_emf);
      auto const & i = coords_face[IX];
      auto const & j = coords_face[IY];

      auto coords_cell = coords_face;

      if (i == 0 and face_xmin_neighbor_is_coarser)
      {
        if (j == block_sizes[IY])
          coords_cell[IY] -= 1;
        constexpr shift_t<dim> shift{ -1, 0 };
        const auto             key_cur = orchard_keys_device(iOct);
        const CellLocation_t   cell_loc{ coords_cell, key_cur, iOct, false };
        const auto             cell_loc_neigh = stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk_cell = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        auto ijk_face = ijk_cell;
        if (j == block_sizes[IY])
          ijk_face[IY] += 1;

        if ((j & 0x1) == 0)
        {
          printf("i=%d j=%d iOct=%d | in=%d jn=%d iOctn=%ld\n",
                 i,
                 j,
                 iOct,
                 ijk_face[IX],
                 ijk_face[IY],
                 iOct_neigh);
        }
      }
      else
      {
        printf("i=%d j=%d iOct=%d\n", i, j, iOct);
      }
    }
  }

  std::list<uint32_t> octs_fine = { 1247, 1264 };

  for (const auto & iOct : octs_fine)
  {
    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(conformal_status(iOct)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    for (int32_t cell_index = 0; cell_index < Kokkos::dim_prod(block_sizes); ++cell_index)
    {
      auto const   coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      auto const & i = coords_cell[IX];
      auto const & j = coords_cell[IY];

      if (i == block_sizes[IX] - 1 and face_xmax_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ 1, 0 };
        const auto             key_cur = orchard_keys_device(iOct);
        const CellLocation_t   cell_loc{ coords_cell, key_cur, iOct, false };
        const auto             cell_loc_neigh = stencil_helper.getNeighLocFiner(cell_loc, shift);

        const auto ijk_cell = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        if ((j & 0x1) == 0)
        {
          printf("i=%d j=%d iOct=%d | in=%d jn=%d iOctn=%ld\n",
                 i,
                 j,
                 iOct,
                 ijk_cell[IX],
                 ijk_cell[IY],
                 iOct_neigh);
        }
      }
      else if (i == 0 and face_xmin_neighbor_is_finer)
      {
        constexpr shift_t<dim> shift{ -1, 0 };
        const auto             key_cur = orchard_keys_device(iOct);
        const CellLocation_t   cell_loc{ coords_cell, key_cur, iOct, false };
        const auto             cell_loc_neigh = stencil_helper.getNeighLocFiner(cell_loc, shift);

        const auto ijk_cell = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        // if ((j & 0x1) == 0)
        {
          printf("i=%d j=%d iOct=%d | in=%d jn=%d iOctn=%ld\n",
                 i,
                 j,
                 iOct,
                 ijk_cell[IX],
                 ijk_cell[IY],
                 iOct_neigh);
        }
      }
      else
      {
        printf("i=%d j=%d iOct=%d\n", i, j, iOct);
      }
    }
  }

} // run_test

} // namespace kalypsso

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
        std::cout << "Example cmdline: \"./test_AMRmesh_stencil_helper --ini test_AMRmesh_stencil_2d.ini \"\n";
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

    if (use_3d)
    {
      // run a 3d test
      kalypsso::run_test<3, kalypsso::DefaultDevice>(par_env, config_map);
    }
    else
    {
      // run a 2d test
      kalypsso::run_test<2, kalypsso::DefaultDevice>(par_env, config_map);
    }


    if (par_env.rank() == 0)
    {
      printf("================================================\n");
      printf("END RUN_TEST\n");
      printf("================================================\n");
    }
  }

  return EXIT_SUCCESS;

} // main
