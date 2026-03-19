// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file StencilHelper_test.cpp
 *
 * unit test for class StencilHelper.
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

  // rebuild the hashmap (must be done after AMR cycle)
  constexpr bool on_device = true;
  mesh_map->update_hashmap(on_device);
  auto amr_hashmap_device = mesh_map->hashmap();
  mesh_map->update_conformal_status();
  auto conformal_status = mesh_map->conformal_status();


  auto amr_mesh_info = initial_amr_setup.amr_mesh_info();

  // for testing DataArrayBlock
  auto userdata_block_cell = initial_amr_setup.setup_initial_data_block_new(orchard_keys_device);

  // for testing FaceDataArrayBlock_t
  auto userdata_block_face =
    initial_amr_setup.setup_initial_data_block_face(orchard_keys_device, false);

  // for testing flux
  auto userdata_block_face_x =
    initial_amr_setup.setup_initial_data_block_flux(orchard_keys_device, IX);

  // for testing flux
  auto userdata_block_face_y =
    initial_amr_setup.setup_initial_data_block_flux(orchard_keys_device, IY);

  {
    std::string filename = dim == 2 ? "StencilHelper_test_2d" : "StencilHelper_test_3d";

    // provide mapping between variables Id and variables names
    const Hydro_t model(dim);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename, userdata_block_cell, config_map, amr_mesh, model);

    DataWriter<dim, device_t, Hydro_t>::save(
      filename + "_face", userdata_block_face, config_map, amr_mesh, "facedata");
  }

  auto stencil_helper = StencilHelper_t(
    amr_hashmap_device, orchard_keys_device, block_sizes, brick_sizes, is_brick_periodic);


  if constexpr (dim == 2)
  {
    {
      uint32_t iOct = 1235;

      const auto face_xmin_neighbor_is_coarser =
        conformal_face_status_t<dim>::face_xmin(conformal_status(iOct)) ==
        conformal_neighbor_status::NEIGHBOR_IS_COARSER;

      EXPECT_EQ(face_xmin_neighbor_is_coarser, false);

      const auto face_xmax_neighbor_is_coarser =
        conformal_face_status_t<dim>::face_xmax(conformal_status(iOct)) ==
        conformal_neighbor_status::NEIGHBOR_IS_COARSER;

      EXPECT_EQ(face_xmax_neighbor_is_coarser, true);

      {
        uint32_t   cell_index = 15; // i=3, j=3 => cell_index = 3 + 4 * 3
        auto const coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);

        constexpr shift_t<2> shift{ 0, 1 };
        const auto           key_cur = orchard_keys_device(iOct);
        const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
        const auto           cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc, shift);

        EXPECT_EQ(cell_loc_neigh.iOct, 1264);
      }

      {
        uint32_t   cell_index = 7; // i=3, j=1 => cell_index = 3 + 4 * 1
        auto const coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);

        constexpr shift_t<2> shift{ 1, 0 };
        const auto           key_cur = orchard_keys_device(iOct);
        const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
        const auto           cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc, shift);

        EXPECT_EQ(cell_loc_neigh.iOct, 1236);
        EXPECT_EQ(cell_loc_neigh.ijk[IX], 0);
        EXPECT_EQ(cell_loc_neigh.ijk[IY], 2);
      }
    }

    {
      uint32_t iOct = 1246;

      const auto face_xmin_neighbor_is_coarser =
        conformal_face_status_t<dim>::face_xmin(conformal_status(iOct)) ==
        conformal_neighbor_status::NEIGHBOR_IS_COARSER;

      EXPECT_EQ(face_xmin_neighbor_is_coarser, true);
    }

    {
      uint32_t iOct = 1265;

      const auto face_ymin_neighbor_is_finer =
        conformal_face_status_t<dim>::face_ymin(conformal_status(iOct)) ==
        conformal_neighbor_status::NEIGHBOR_IS_FINER;

      EXPECT_EQ(face_ymin_neighbor_is_finer, true);

      const auto face_ymax_neighbor_is_coarser =
        conformal_face_status_t<dim>::face_ymax(conformal_status(iOct)) ==
        conformal_neighbor_status::NEIGHBOR_IS_COARSER;

      EXPECT_EQ(face_ymax_neighbor_is_coarser, true);

      {
        uint32_t   cell_index = 1; // i=1, j=0 => cell_index = 1 + 4 * 0
        auto const coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);

        constexpr shift_t<2> shift{ 0, -1 };
        const auto           key_cur = orchard_keys_device(iOct);
        const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
        const auto           cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc, shift);

        EXPECT_EQ(cell_loc_neigh.iOct, 1262);
        EXPECT_EQ(cell_loc_neigh.ijk[IX], 2);
        EXPECT_EQ(cell_loc_neigh.ijk[IY], 2);
      }

      {
        uint32_t   cell_index = 2; // i=2, j=0 => cell_index = 2 + 4 * 0
        auto const coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);

        constexpr shift_t<2> shift{ 0, -1 };
        const auto           key_cur = orchard_keys_device(iOct);
        const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
        const auto           cell_loc_neigh = stencil_helper.getNeighLoc(cell_loc, shift);

        EXPECT_EQ(cell_loc_neigh.iOct, 1263);
        EXPECT_EQ(cell_loc_neigh.ijk[IX], 0);
        EXPECT_EQ(cell_loc_neigh.ijk[IY], 2);
      }
    }

    // checking compute_siblings_average
    {
      uint32_t iOct = 1260;

      const uint32_t       i = 2;
      const uint32_t       j = 0;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = 0;

      const auto average =
        stencil_helper.compute_siblings_average(cell_loc, block_sizes, ivar, userdata_block_cell);

      const auto average_true = (userdata_block_cell(i + 0, j + 0, ivar, iOct) +
                                 userdata_block_cell(i + 1, j + 0, ivar, iOct) +
                                 userdata_block_cell(i + 0, j + 1, ivar, iOct) +
                                 userdata_block_cell(i + 1, j + 1, ivar, iOct)) /
                                4;

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_siblings_average failed - iOct=" << iOct;
    }

    // checking compute_face_siblings_sum - cell data
    {
      uint32_t iOct = 1247;

      const uint32_t       i = 3;
      const uint32_t       j = 0;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = 0;

      const auto sum =
        stencil_helper.compute_face_siblings_sum(cell_loc, ivar, userdata_block_cell, IX);

      const auto sum_true = (userdata_block_cell(i + 0, j + 0, ivar, iOct) +
                             userdata_block_cell(i + 0, j + 1, ivar, iOct));

      EXPECT_NEAR(sum, sum_true, 1e-14)
        << "checking compute_face_siblings_sum failed - iOct=" << iOct;
    }
    {
      uint32_t iOct = 1260;

      const uint32_t       i = 1;
      const uint32_t       j = 2;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = 0;

      const auto sum =
        stencil_helper.compute_face_siblings_sum(cell_loc, ivar, userdata_block_cell, IY);

      const auto sum_true = (userdata_block_cell(i + 0, j + 0, ivar, iOct) +
                             userdata_block_cell(i - 1, j + 0, ivar, iOct));

      EXPECT_NEAR(sum, sum_true, 1e-14)
        << "checking compute_face_siblings_sum failed - iOct=" << iOct;
    }

    // checking compute_face_siblings_sum - flux data
    {
      uint32_t iOct = 1260;

      const uint32_t       i = 0;
      const uint32_t       j = 2;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = 0;

      EXPECT_EQ(userdata_block_face_x.is_flux_array(block_sizes, IX), true);

      const bool use_right_face = false;

      const auto sum = stencil_helper.compute_face_siblings_sum(
        cell_loc, ivar, userdata_block_face_x, use_right_face);

      auto ijk_face = cell_loc.ijk;
      if (use_right_face)
      {
        ijk_face[IX] += 1;
      }
      const auto face_multiindex = to_face_multiindex<dim>(ijk_face, IX);

      const auto sum_true = (userdata_block_face_x(ijk_face[IX], ijk_face[IY] + 0, ivar, iOct) +
                             userdata_block_face_x(ijk_face[IX], ijk_face[IY] + 1, ivar, iOct));

      EXPECT_NEAR(sum, sum_true, 1e-14) << "checking compute_face_siblings_sum failed - left face";
    }

    {
      uint32_t iOct = 1261;

      const uint32_t       i = 3;
      const uint32_t       j = 1;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = 0;

      EXPECT_EQ(userdata_block_face_x.is_flux_array(block_sizes, IX), true);

      const bool use_right_face = true;

      const auto sum = stencil_helper.compute_face_siblings_sum(
        cell_loc, ivar, userdata_block_face_x, use_right_face);

      auto ijk_face = cell_loc.ijk;
      {
        ijk_face[IX] += 1;
      }

      const auto face_multiindex = to_face_multiindex<dim>(ijk_face, IX);

      const auto sum_true = (userdata_block_face_x(ijk_face[IX], ijk_face[IY] + 0, ivar, iOct) +
                             userdata_block_face_x(ijk_face[IX], ijk_face[IY] - 1, ivar, iOct));

      EXPECT_NEAR(sum, sum_true, 1e-14) << "checking compute_face_siblings_sum failed - right face";
    }

    // testing StencilHelper with FaceDataArrayBlock
    {
      uint32_t iOct = 1216;

      const uint32_t       i = 1;
      const uint32_t       j = 1;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IX;
      const auto           ivar_face = face_type_t::LEFT;

      const auto average = stencil_helper.compute_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      auto ijk_face = cell_loc.ijk;
      if (ivar_face == face_type_t::RIGHT)
      {
        ijk_face[ivar] += 1;
      }

      const auto face_multiindex = to_face_multiindex<dim>(ijk_face, ivar);

      const auto average_true = (userdata_block_face(ijk_face[IX], ijk_face[IY] + 0, ivar, iOct) +
                                 userdata_block_face(ijk_face[IX], ijk_face[IY] - 1, ivar, iOct)) /
                                2;

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - left face";
    }

    {
      uint32_t iOct = 1216;

      const uint32_t       i = 2;
      const uint32_t       j = 3;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IY;
      const auto           ivar_face = face_type_t::RIGHT;

      const auto average = stencil_helper.compute_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      auto ijk_face = cell_loc.ijk;
      if (ivar_face == face_type_t::RIGHT)
      {
        ijk_face[ivar] += 1;
      }

      const auto face_multiindex = to_face_multiindex<dim>(ijk_face, ivar);

      const auto average_true = (userdata_block_face(ijk_face[IX] + 0, ijk_face[IY], ivar, iOct) +
                                 userdata_block_face(ijk_face[IX] + 1, ijk_face[IY], ivar, iOct)) /
                                2;

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - right face";
    }

    {
      uint32_t iOct = 1216;

      const uint32_t       i = 1;
      const uint32_t       j = 1;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IZ;
      const auto           ivar_face = face_type_t::RIGHT;

      const auto average = stencil_helper.compute_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      auto ijk_face = cell_loc.ijk;

      const auto face_multiindex = to_face_multiindex<dim>(ijk_face, ivar);

      const auto average_true =
        (userdata_block_face(ijk_face[IX] - 1, ijk_face[IY] - 1, ivar, iOct) +
         userdata_block_face(ijk_face[IX] + 0, ijk_face[IY] - 1, ivar, iOct) +
         userdata_block_face(ijk_face[IX] - 1, ijk_face[IY] + 0, ivar, iOct) +
         userdata_block_face(ijk_face[IX] + 0, ijk_face[IY] + 0, ivar, iOct)) /
        4;

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - ivar = IZ ";
    }

    // ===================================================================
    // testing compute_external_face_siblings_average
    // ===================================================================
    {
      uint32_t iOct = 1216;

      // i and j coordinates must be all even (eldest sibling)
      const uint32_t       i = 2;
      const uint32_t       j = 2;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IX;
      const auto           ivar_face = face_type_t::LEFT;

      const auto average = stencil_helper.compute_external_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      // clang-format off
      const auto average_true = (userdata_block_face(i, j + 0, ivar, iOct) +
                                 userdata_block_face(i, j + 1, ivar, iOct)) /
                                2;
      // clang-format on

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - left face";
    }

    {
      uint32_t iOct = 1216;

      // i and j coordinates must be all even (eldest sibling)
      const uint32_t       i = 2;
      const uint32_t       j = 0;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IY;
      const auto           ivar_face = face_type_t::LEFT;

      const auto average = stencil_helper.compute_external_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      // clang-format off
      const auto average_true = (userdata_block_face(i + 0, j + 0, ivar, iOct) +
                                 userdata_block_face(i + 1, j + 0, ivar, iOct)) /
                                2;
      // clang-format on

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - left face";
    }

    {
      uint32_t iOct = 1216;

      // i and j coordinates must be all even (eldest sibling)
      const uint32_t       i = 0;
      const uint32_t       j = 2;
      const uint32_t       cell_index = i + 4 * j;
      auto const           coords_cell = cellindex_to_coord<dim>(cell_index, block_sizes);
      const auto           key_cur = orchard_keys_device(iOct);
      const CellLocation_t cell_loc{ coords_cell, key_cur, iOct, false };
      const auto           ivar = IY;
      const auto           ivar_face = face_type_t::RIGHT;

      const auto average = stencil_helper.compute_external_face_siblings_average(
        cell_loc, ivar, ivar_face, userdata_block_face);

      // clang-format off
      const auto average_true = (userdata_block_face(i + 0, j + 2, ivar, iOct) +
                                 userdata_block_face(i + 1, j + 2, ivar, iOct)) /
                                2;
      // clang-format on

      EXPECT_NEAR(average, average_true, 1e-14)
        << "checking compute_face_siblings_average failed - left face";
    }

    // testing getBorderEdgeLocSymmetric
    {
      uint32_t                   iOct = 2332;
      const edge_multiindex_t<2> edge_indexes{ 0, 3, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      EXPECT_EQ(edge_loc_neigh.iOct == 2163, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 2, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 2332;
      const edge_multiindex_t<2> edge_indexes{ 0, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      EXPECT_EQ(edge_loc_neigh.iOct == 1655, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1670;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      EXPECT_EQ(edge_loc_neigh.iOct == 2333, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 1, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      EXPECT_EQ(edge_loc_neigh.iOct == 1666, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1666;
      const edge_multiindex_t<2> edge_indexes{ 3, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      // should return "self" since neighbor is coarser
      EXPECT_EQ(edge_loc_neigh.iOct == 1666, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 3, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1665;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh = stencil_helper.getBorderEdgeLocSymmetric(edge_loc);

      // should return "self" since neighbor is coarser
      EXPECT_EQ(edge_loc_neigh.iOct == 1667, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR1);

      EXPECT_EQ(edge_loc_neigh.iOct == 1647, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR2);

      EXPECT_EQ(edge_loc_neigh.iOct == 1666, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 0, 3, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR1);

      EXPECT_EQ(edge_loc_neigh.iOct == 1647, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 2, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 2, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR1);

      // probably a bit strange, if we shift along the border (stay inside current block), but edge
      // index doesn't change
      // in normal operation we should not face that case
      EXPECT_EQ(edge_loc_neigh.iOct == 1664, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR1);

      EXPECT_EQ(edge_loc_neigh.iOct == 1278, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 1, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIR2);

      EXPECT_EQ(edge_loc_neigh.iOct == 1281, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 2, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_neigh =
        stencil_helper.getBorderEdgeLocSymmetric(edge_loc, EdgeNormalType::DIAGONAL);

      EXPECT_EQ(edge_loc_neigh.iOct == 1282, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_neigh.ijk[2] == 2, true);
    }

    // ===================================================================
    // testing getEdgeSiblingLoc
    // ===================================================================
    {
      uint32_t                   iOct = 1667;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_01);

      EXPECT_EQ(edge_loc_sibling.iOct == 1664, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1667;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1665, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1667;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1670, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    /////////////////
    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 2, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_10);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1664, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 2, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1666, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 2, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1667, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    /////////////////
    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 3, 1, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_00);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1664, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 1, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 3, 1, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1664, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 1, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1664;
      const edge_multiindex_t<2> edge_indexes{ 3, 1, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1664, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 1, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    /////////////////
    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 4, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_11);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1626, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 4, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1282, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1279;
      const edge_multiindex_t<2> edge_indexes{ 4, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1632, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    /////////////////
    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_10);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1712, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1714, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1728, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    /////////////////
    {
      uint32_t                   iOct = 594;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_01);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 592, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 594;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1287, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 594;
      const edge_multiindex_t<2> edge_indexes{ 4, 0, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1293, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    ///////////////// should test invalid edge lcations
    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 3, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_00);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1712, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, false);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 3, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1725, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 3, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1725;
      const edge_multiindex_t<2> edge_indexes{ 0, 3, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1712, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, false);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    ///////////////// should test invalid edge lcations
    {
      uint32_t                   iOct = 1726;
      const edge_multiindex_t<2> edge_indexes{ 3, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      EXPECT_EQ(get_CellEdgeLocation<2>(edge_loc, block_sizes), EDGE_10);

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(edge_loc_sibling.iOct == 1726, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, true);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1726;
      const edge_multiindex_t<2> edge_indexes{ 3, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 1728, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IY] == 3, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, false);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

    {
      uint32_t                   iOct = 1726;
      const edge_multiindex_t<2> edge_indexes{ 3, 4, IZ };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<2>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 1728, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      // EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.is_valid, false);
      EXPECT_EQ(edge_loc_sibling.ijk[2] == 2, true);
    }

  } // dim ==2
  else if constexpr (dim == 3)
  {
    // ===================================================================
    // testing getEdgeSiblingLoc
    // ===================================================================
    {
      uint32_t                   iOct = 29;
      const edge_multiindex_t<3> edge_indexes{ 0, 0, 4, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(get_CellEdgeLocation<3>(edge_loc, block_sizes), EDGE_10);

      EXPECT_EQ(edge_loc_sibling.iOct == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    {
      uint32_t                   iOct = 29;
      const edge_multiindex_t<3> edge_indexes{ 0, 0, 4, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 7, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    {
      uint32_t                   iOct = 29;
      const edge_multiindex_t<3> edge_indexes{ 0, 0, 4, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 56, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    ////////////////////
    {
      uint32_t                   iOct = 106;
      const edge_multiindex_t<3> edge_indexes{ 2, 2, 0, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(get_CellEdgeLocation<3>(edge_loc, block_sizes), EDGE_00);

      EXPECT_EQ(edge_loc_sibling.iOct == 79, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    {
      uint32_t                   iOct = 106;
      const edge_multiindex_t<3> edge_indexes{ 2, 2, 0, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 81, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    {
      uint32_t                   iOct = 106;
      const edge_multiindex_t<3> edge_indexes{ 2, 2, 0, IX };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 106, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IX, true);
    }

    ////////////////////
    {
      uint32_t                   iOct = 126;
      const edge_multiindex_t<3> edge_indexes{ 4, 3, 4, IY };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 0);

      EXPECT_EQ(get_CellEdgeLocation<3>(edge_loc, block_sizes), EDGE_11);

      EXPECT_EQ(edge_loc_sibling.iOct == 147, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IY, true);
    }

    {
      uint32_t                   iOct = 126;
      const edge_multiindex_t<3> edge_indexes{ 4, 3, 4, IY };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 1);

      EXPECT_EQ(edge_loc_sibling.iOct == 144, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 4, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 3, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IY, true);
    }

    {
      uint32_t                   iOct = 126;
      const edge_multiindex_t<3> edge_indexes{ 4, 3, 4, IY };
      const auto                 key_cur = orchard_keys_device(iOct);
      const EdgeLocation<3>      edge_loc{ edge_indexes, key_cur, iOct, false };

      const auto edge_loc_sibling = stencil_helper.getEdgeSiblingLoc(edge_loc, 2);

      EXPECT_EQ(edge_loc_sibling.iOct == 160, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IX] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IY] == 2, true);
      EXPECT_EQ(edge_loc_sibling.ijk[IZ] == 0, true);
      EXPECT_EQ(edge_loc_sibling.ijk[dim] == IY, true);
    }


  } // dim == 3

} // run_test

// ====================================================================
// ====================================================================
TEST(kalypsso_shared, stencil_helper)
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

// ====================================================================
// ====================================================================
TEST(kalypsso_shared, test_edge_utils)
{
  // 2d - testing EdgeUtils
  {
    const auto shift0 = EdgeUtils::edge_neighbor_shift<2>(EDGE_10, 0, IZ);
    EXPECT_EQ(shift0[IX] == -1, true);
    EXPECT_EQ(shift0[IY] == 0, true);

    const auto shift1 = EdgeUtils::edge_neighbor_shift<2>(EDGE_11, 2, IZ);
    EXPECT_EQ(shift1[IX] == 1, true);
    EXPECT_EQ(shift1[IY] == 1, true);

    const auto shift2 = EdgeUtils::edge_neighbor_shift<2>(EDGE_00, 1, IZ);
    EXPECT_EQ(shift2[IX] == 0, true);
    EXPECT_EQ(shift2[IY] == -1, true);
  }

  // 3d - testing EdgeUtils
  {
    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_10, 0, IZ);
      EXPECT_EQ(shift0[IX] == -1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_10, 0, IX);
      EXPECT_EQ(shift0[IX] == 0, true);
      EXPECT_EQ(shift0[IY] == -1, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_10, 0, IY);
      EXPECT_EQ(shift0[IX] == -1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_01, 2, IZ);
      EXPECT_EQ(shift0[IX] == 1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_01, 2, IX);
      EXPECT_EQ(shift0[IX] == 0, true);
      EXPECT_EQ(shift0[IY] == 1, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_01, 2, IY);
      EXPECT_EQ(shift0[IX] == 1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_00, 1, IZ);
      EXPECT_EQ(shift0[IX] == 0, true);
      EXPECT_EQ(shift0[IY] == -1, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_00, 1, IX);
      EXPECT_EQ(shift0[IX] == 0, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == -1, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_00, 0, IY);
      EXPECT_EQ(shift0[IX] == -1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == -1, true);
    }

    {
      const auto shift0 = EdgeUtils::edge_neighbor_shift<3>(EDGE_11, 0, IY);
      EXPECT_EQ(shift0[IX] == 1, true);
      EXPECT_EQ(shift0[IY] == 0, true);
      EXPECT_EQ(shift0[IZ] == 0, true);
    }
  }
}

} // namespace kalypsso
