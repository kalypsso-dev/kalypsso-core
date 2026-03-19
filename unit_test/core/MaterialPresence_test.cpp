// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kalypsso/core/MaterialPresence.h>
#include <kalypsso/core/MultiMatFillBlockGhostCells.h>
#include <kalypsso/core/UserDataRemapper.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/AMRContext.h>

#include "gtest/gtest.h"
#include "../main_kalypsso_unittest.h"

namespace kalypsso
{

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, MaterialPresence)
{
  DataArrayUtils::set_growth_rate(KALYPSSO_NUM(2.5));

  using MP_t = MaterialPresenceView<HostDevice>;

  MP_t mats("mat pres", 120, 3);

  EXPECT_EQ(mats.num_materials(1), 0);
  EXPECT_EQ(mats.size(), 3);
  EXPECT_EQ(mats.capacity(), 15);

  mats.set(1, 25);
  mats.set(1, 44);
  mats.set(1, 96);

  EXPECT_TRUE(mats.get(1, 25));
  EXPECT_TRUE(mats.get(1, 44));
  EXPECT_TRUE(mats.get(1, 96));

  EXPECT_FALSE(mats.get(1, 30));
  EXPECT_FALSE(mats.get(1, 100));

  EXPECT_EQ(mats.material_num(1, 0), 25);
  EXPECT_EQ(mats.material_num(1, 1), 44);
  EXPECT_EQ(mats.material_num(1, 2), 96);

  EXPECT_EQ(mats.material_index(1, 25), 0);
  EXPECT_EQ(mats.material_index(1, 44), 1);
  EXPECT_EQ(mats.material_index(1, 96), 2);

  EXPECT_EQ(mats.num_materials(1), 3);

  mats.set(0, 1);
  mats.set(0, 44);
  mats.set(0, 100);

  MP_t::copy(mats, 2, mats, 0);
  MP_t::update(mats, 2, mats, 1);
  EXPECT_TRUE(mats.get(2, 1));
  EXPECT_TRUE(mats.get(2, 25));
  EXPECT_TRUE(mats.get(2, 44));
  EXPECT_TRUE(mats.get(2, 96));
  EXPECT_TRUE(mats.get(2, 100));
  EXPECT_FALSE(mats.get(2, 30));

  EXPECT_EQ(mats.num_materials(2), 5);

  mats.unset(2, 44);
  EXPECT_FALSE(mats.get(2, 44));
  EXPECT_FALSE(mats.get(2, 30));
  EXPECT_EQ(mats.num_materials(2), 4);

  Kokkos::View<uint32_t *, HostDevice> num_vars("num vars", 3);
  mats.compute_num_vars(5, num_vars);

  EXPECT_EQ(num_vars(0), 15);
  EXPECT_EQ(num_vars(1), 15);
  EXPECT_EQ(num_vars(2), 20);

  mats.resets();
  EXPECT_EQ(mats.num_materials(0), 0);
  EXPECT_EQ(mats.num_materials(1), 0);
  EXPECT_EQ(mats.num_materials(2), 0);
}

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, MaterialPresence_fill_ghosts_2d)
{
  static char         null_char = '\0';
  static char *       empty_string = &null_char;
  const ParallelEnv & par_env = **g_par_env_ptr;
  ConfigMap           config_map(empty_string, 1);

  config_map.setInteger("amr", "level_min", 0);
  config_map.setInteger("amr", "level_max", 2);
  config_map.setString("amr", "connectivity", "brick");

  config_map.setInteger("p4est_connectivity", "nbrick_x", 2);
  config_map.setInteger("p4est_connectivity", "nbrick_y", 2);

  AMRmesh<2>             amr_mesh(par_env, config_map);
  MeshMap<2, HostDevice> amr_mesh_map(config_map, par_env);

  auto rebuild_forest = [&]() {
    amr_mesh.reset_ghost();
    amr_mesh_map.compute_outside_quad_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_hashmap(amr_mesh.forest(), amr_mesh.ghost(), true);
    amr_mesh_map.update_mirror_orchard_keys(amr_mesh.ghost());
    amr_mesh_map.update_amr_mesh_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_conformal_status();
  };

  rebuild_forest();

  {
    const auto                num_quadrants = amr_mesh.local_num_quadrants();
    AMRContext<2, HostDevice> amr_context(num_quadrants);
    const auto                flags = amr_context.m_amrflags_h;

    for (uint32_t i_oct = 0; i_oct < num_quadrants; i_oct++)
      if (i_oct == 0 || i_oct == 2)
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_REFINE;
      else
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_NOTHING;

    amr_context.adapt_mesh(amr_mesh.forest());
  }

  rebuild_forest();

  {
    const auto                num_quadrants = amr_mesh.local_num_quadrants();
    AMRContext<2, HostDevice> amr_context(num_quadrants);
    const auto                flags = amr_context.m_amrflags_h;

    for (uint32_t i_oct = 0; i_oct < num_quadrants; i_oct++)
      if (i_oct == 5)
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_REFINE;
      else
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_NOTHING;

    amr_context.adapt_mesh(amr_mesh.forest());
  }

  rebuild_forest();

  const auto nb_owned = amr_mesh.local_num_quadrants();
  const auto nb_outside = amr_mesh_map.get_amr_mesh_info().local_num_quadrants_outside();

  using MatPresView_t = MaterialPresenceView<HostDevice>;

  MatPresView_t mat_pres_src("mat pres src", 3, nb_owned + nb_outside);
  MatPresView_t mat_pres_dst("mat pres dst", 3, nb_owned);

  {
    for (uint32_t i_oct = 0; i_oct < nb_owned; i_oct++)
      // clang-format off
        switch (i_oct)
        {
          case 0: case 1: case 2: case 3: case 5: case 6: case 7: case 9:
            mat_pres_src.set(i_oct, 2); break;
          case 4: case 8: case 10: case 11:
            mat_pres_src.set(i_oct, 1); break;
          case 12:
            mat_pres_src.set(i_oct, 0); break;
        }
    // clang-format on
  };

  const auto keys = amr_mesh_map.orchard_keys();
  const auto hashmap = amr_mesh_map.hashmap();
  for (uint32_t i_oct = nb_outside; i_oct < nb_owned + nb_outside; i_oct++)
  {
    const auto key_outside = keys(i_oct);

    // Get the direction towards inside the domain
    auto outside_normal =
      orchard_key_t<2>::get_outside_normal(key_outside, { 2, 2 }, { true, true });
    outside_normal[IX] *= orchard_key_t<2>::is_touching_face_X(key_outside);
    outside_normal[IY] *= orchard_key_t<2>::is_touching_face_Y(key_outside);

    // Get the inside key
    auto key_inside = orchard_key_t<2>::get_neighbor_key_same_level(
      key_outside, outside_normal, { 2, 2 }, { false, false });
    orchard_key_t<2>::reset_outside_bits(key_inside);

    const auto i_oct_inside = hashmap.value_at(hashmap.find(key_inside));
    MatPresView_t::copy(mat_pres_src, i_oct, mat_pres_src, i_oct_inside);
  }


  MultiMatFillBlockGhostCellsMatPresence<2, HostDevice>::apply(mat_pres_src,
                                                               amr_mesh_map.hashmap(),
                                                               amr_mesh_map.orchard_keys(),
                                                               { 2, 2 },
                                                               { false, false },
                                                               0,
                                                               nb_owned,
                                                               mat_pres_dst);

  EXPECT_FALSE(mat_pres_dst.get(0, 0));
  EXPECT_FALSE(mat_pres_dst.get(0, 1));
  EXPECT_TRUE(mat_pres_dst.get(0, 2));

  EXPECT_FALSE(mat_pres_dst.get(1, 0));
  EXPECT_TRUE(mat_pres_dst.get(1, 1));
  EXPECT_TRUE(mat_pres_dst.get(1, 2));

  EXPECT_FALSE(mat_pres_dst.get(2, 0));
  EXPECT_FALSE(mat_pres_dst.get(2, 1));
  EXPECT_TRUE(mat_pres_dst.get(2, 2));

  EXPECT_TRUE(mat_pres_dst.get(3, 0));
  EXPECT_TRUE(mat_pres_dst.get(3, 1));
  EXPECT_TRUE(mat_pres_dst.get(3, 2));

  EXPECT_TRUE(mat_pres_dst.get(4, 0));
  EXPECT_TRUE(mat_pres_dst.get(4, 1));
  EXPECT_TRUE(mat_pres_dst.get(4, 2));

  EXPECT_FALSE(mat_pres_dst.get(5, 0));
  EXPECT_TRUE(mat_pres_dst.get(5, 1));
  EXPECT_TRUE(mat_pres_dst.get(5, 2));

  EXPECT_FALSE(mat_pres_dst.get(6, 0));
  EXPECT_TRUE(mat_pres_dst.get(6, 1));
  EXPECT_TRUE(mat_pres_dst.get(6, 2));

  EXPECT_FALSE(mat_pres_dst.get(7, 0));
  EXPECT_TRUE(mat_pres_dst.get(7, 1));
  EXPECT_TRUE(mat_pres_dst.get(7, 2));

  EXPECT_FALSE(mat_pres_dst.get(8, 0));
  EXPECT_TRUE(mat_pres_dst.get(8, 1));
  EXPECT_TRUE(mat_pres_dst.get(8, 2));

  EXPECT_TRUE(mat_pres_dst.get(9, 0));
  EXPECT_TRUE(mat_pres_dst.get(9, 1));
  EXPECT_TRUE(mat_pres_dst.get(9, 2));

  EXPECT_FALSE(mat_pres_dst.get(10, 0));
  EXPECT_TRUE(mat_pres_dst.get(10, 1));
  EXPECT_TRUE(mat_pres_dst.get(10, 2));

  EXPECT_TRUE(mat_pres_dst.get(11, 0));
  EXPECT_TRUE(mat_pres_dst.get(11, 1));
  EXPECT_TRUE(mat_pres_dst.get(11, 2));

  EXPECT_TRUE(mat_pres_dst.get(12, 0));
  EXPECT_TRUE(mat_pres_dst.get(12, 1));
  EXPECT_TRUE(mat_pres_dst.get(12, 2));
}

// ================================================================================================
// ================================================================================================
TEST(kalypsso_shared_test, MaterialPresence_amr)
{
  static char         null_char = '\0';
  static char *       empty_string = &null_char;
  const ParallelEnv & par_env = **g_par_env_ptr;
  ConfigMap           config_map(empty_string, 1);

  config_map.setInteger("amr", "level_min", 0);
  config_map.setInteger("amr", "level_max", 1);
  config_map.setString("amr", "connectivity", "brick");

  config_map.setInteger("p4est_connectivity", "nbrick_x", 2);
  config_map.setInteger("p4est_connectivity", "nbrick_y", 2);

  AMRmesh<2>             amr_mesh(par_env, config_map);
  MeshMap<2, HostDevice> amr_mesh_map(config_map, par_env);

  using MatPresView_t = MaterialPresenceView<HostDevice>;

  MatPresView_t mat_pres_old("mat pres A", 3, 0);
  MatPresView_t mat_pres_new("mat pres B", 3, 0);

  auto rebuild_forest = [&]() {
    amr_mesh.reset_ghost();
    amr_mesh_map.compute_outside_quad_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_hashmap(amr_mesh.forest(), amr_mesh.ghost(), true);
    amr_mesh_map.update_mirror_orchard_keys(amr_mesh.ghost());
    amr_mesh_map.update_amr_mesh_info(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_orchard_keys(amr_mesh.forest(), amr_mesh.ghost());
    amr_mesh_map.update_conformal_status();
  };

  rebuild_forest();

  {
    const auto                num_quadrants = amr_mesh.local_num_quadrants();
    AMRContext<2, HostDevice> amr_context(num_quadrants);
    const auto                flags = amr_context.m_amrflags_d;

    mat_pres_old.resize(num_quadrants);
    mat_pres_old.resets();

    for (uint32_t i_oct = 0; i_oct < num_quadrants; i_oct++)
      switch (i_oct)
      {
        case 0:
          flags(i_oct) = AMRContextBase::KALYPSSO_DO_REFINE;
          mat_pres_old.set(i_oct, 0);
          break;
        case 1:
          flags(i_oct) = AMRContextBase::KALYPSSO_DO_NOTHING;
          mat_pres_old.set(i_oct, 1);
          break;
        case 2:
          flags(i_oct) = AMRContextBase::KALYPSSO_DO_REFINE;
          mat_pres_old.set(i_oct, 1);
          break;
        case 3:
          flags(i_oct) = AMRContextBase::KALYPSSO_DO_NOTHING;
          mat_pres_old.set(i_oct, 2);
          break;
      };


    amr_context.adapt_mesh(amr_mesh.forest());
  }

  {
    const auto old_hashmap = amr_mesh_map.hashmap_clone();
    const auto old_keys = amr_mesh_map.orchard_keys();

    rebuild_forest();

    const auto new_keys = amr_mesh_map.orchard_keys();

    UserDataRemapper<2, HostDevice> remapper(
      old_hashmap, new_keys, old_keys, amr_mesh.local_num_quadrants(), { 2, 2 }, config_map);
    mat_pres_new.resize(amr_mesh.local_num_quadrants());
    remapper.remap_material_presence({}, mat_pres_old, mat_pres_new);
    my_swap(mat_pres_old, mat_pres_new);

    EXPECT_TRUE(mat_pres_old.get(0, 0));
    EXPECT_FALSE(mat_pres_old.get(0, 1));
    EXPECT_FALSE(mat_pres_old.get(0, 2));

    EXPECT_TRUE(mat_pres_old.get(1, 0));
    EXPECT_FALSE(mat_pres_old.get(1, 1));
    EXPECT_FALSE(mat_pres_old.get(1, 2));

    EXPECT_TRUE(mat_pres_old.get(2, 0));
    EXPECT_FALSE(mat_pres_old.get(2, 1));
    EXPECT_FALSE(mat_pres_old.get(2, 2));

    EXPECT_TRUE(mat_pres_old.get(3, 0));
    EXPECT_FALSE(mat_pres_old.get(3, 1));
    EXPECT_FALSE(mat_pres_old.get(3, 2));

    EXPECT_FALSE(mat_pres_old.get(4, 0));
    EXPECT_TRUE(mat_pres_old.get(4, 1));
    EXPECT_FALSE(mat_pres_old.get(4, 2));

    EXPECT_FALSE(mat_pres_old.get(5, 0));
    EXPECT_TRUE(mat_pres_old.get(5, 1));
    EXPECT_FALSE(mat_pres_old.get(5, 2));

    EXPECT_FALSE(mat_pres_old.get(6, 0));
    EXPECT_TRUE(mat_pres_old.get(6, 1));
    EXPECT_FALSE(mat_pres_old.get(6, 2));

    EXPECT_FALSE(mat_pres_old.get(7, 0));
    EXPECT_TRUE(mat_pres_old.get(7, 1));
    EXPECT_FALSE(mat_pres_old.get(7, 2));

    EXPECT_FALSE(mat_pres_old.get(8, 0));
    EXPECT_TRUE(mat_pres_old.get(8, 1));
    EXPECT_FALSE(mat_pres_old.get(8, 2));

    EXPECT_FALSE(mat_pres_old.get(9, 0));
    EXPECT_FALSE(mat_pres_old.get(9, 1));
    EXPECT_TRUE(mat_pres_old.get(9, 2));
  }

  {
    const auto                num_quadrants = amr_mesh.local_num_quadrants();
    AMRContext<2, HostDevice> amr_context(num_quadrants);
    const auto                flags = amr_context.m_amrflags_d;

    mat_pres_old.set(5, 0);

    for (uint32_t i_oct = 0; i_oct < num_quadrants; i_oct++)
      if (i_oct >= 5 && i_oct <= 8)
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_COARSEN;
      else
        flags(i_oct) = AMRContextBase::KALYPSSO_DO_NOTHING;

    amr_context.adapt_mesh(amr_mesh.forest());
  }

  {
    const auto old_hashmap = amr_mesh_map.hashmap_clone();
    const auto old_keys = amr_mesh_map.orchard_keys();

    rebuild_forest();

    const auto new_keys = amr_mesh_map.orchard_keys();

    UserDataRemapper<2, HostDevice> remapper(
      old_hashmap, new_keys, old_keys, amr_mesh.local_num_quadrants(), { 2, 2 }, config_map);
    mat_pres_new.resize(amr_mesh.local_num_quadrants());
    remapper.remap_material_presence({}, mat_pres_old, mat_pres_new);
    my_swap(mat_pres_old, mat_pres_new);

    EXPECT_TRUE(mat_pres_old.get(0, 0));
    EXPECT_FALSE(mat_pres_old.get(0, 1));
    EXPECT_FALSE(mat_pres_old.get(0, 2));

    EXPECT_TRUE(mat_pres_old.get(1, 0));
    EXPECT_FALSE(mat_pres_old.get(1, 1));
    EXPECT_FALSE(mat_pres_old.get(1, 2));

    EXPECT_TRUE(mat_pres_old.get(2, 0));
    EXPECT_FALSE(mat_pres_old.get(2, 1));
    EXPECT_FALSE(mat_pres_old.get(2, 2));

    EXPECT_TRUE(mat_pres_old.get(3, 0));
    EXPECT_FALSE(mat_pres_old.get(3, 1));
    EXPECT_FALSE(mat_pres_old.get(3, 2));

    EXPECT_FALSE(mat_pres_old.get(4, 0));
    EXPECT_TRUE(mat_pres_old.get(4, 1));
    EXPECT_FALSE(mat_pres_old.get(4, 2));

    EXPECT_TRUE(mat_pres_old.get(5, 0));
    EXPECT_TRUE(mat_pres_old.get(5, 1));
    EXPECT_FALSE(mat_pres_old.get(5, 2));

    EXPECT_FALSE(mat_pres_old.get(6, 0));
    EXPECT_FALSE(mat_pres_old.get(6, 1));
    EXPECT_TRUE(mat_pres_old.get(6, 2));
  }
}

} // namespace kalypsso
